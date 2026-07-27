import json
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.trainer_utils import is_main_process
from transformers import TrainerCallback
import pandas as pd
import anndata as ad
import scanpy as sc
import polars as pl
from tqdm import tqdm

import wandb

from utils.dataset import collate_fn_infer_perturb_vq
from utils.distribution import NegativeBinomial


class PerturbGenerationEvalCallback(TrainerCallback):
    """
    At every eval_step_per steps, generate perturbed cells on testA/testB,
    decode VQ codes back to expression, and compute cell_eval metrics.
    Generation is distributed across all GPUs; metric computation runs on rank 0.
    """

    METRICS_TO_LOG = [
        'de_direction_match', 'de_spearman_lfc_sig', 'pearson_delta',
        'mae', 'discrimination_score_l2',
    ]

    def __init__(self, eval_datasets_gen, vq_model, tokenizer, ignore_ids,
                 size_factor_val, reference_gene, max_new_tokens,
                 eval_batch_size, eval_step_per, eval_before_train=False,
                 sample_size=None):
        self.eval_datasets_gen = eval_datasets_gen
        self.sample_size = sample_size
        self.vq_model = vq_model
        self.tokenizer = tokenizer
        self.ignore_ids = ignore_ids
        self.size_factor_val = size_factor_val
        self.reference_gene = reference_gene
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        self.eval_step_per = eval_step_per
        self.eval_before_train = eval_before_train
        self._did_eval_before_train = False
        self._gt_expr_cache = {}

    # ---------- VQ code -> expression count ----------
    @torch.no_grad()
    def _decode_vq_to_expr(self, generated_ids_batch, prefix_len):
        num_code = self.vq_model.num_code
        expressions = []
        valid_mask = []

        for gen_ids in generated_ids_batch:
            vq_tokens = gen_ids[prefix_len:]
            decoded_str = self.tokenizer.decode(vq_tokens[:num_code])
            parts = decoded_str.split('##')
            try:
                code_indices = [int(p) for p in parts[:num_code]]
            except (ValueError, IndexError):
                valid_mask.append(False)
                expressions.append(np.zeros(len(self.reference_gene)))
                continue

            codes = self.vq_model.quantize.embedding.weight[code_indices]
            quant = codes.reshape(-1)
            logits = self.vq_model.decoder(quant)
            expressions.append(logits.cpu())
            valid_mask.append(True)

        if not any(valid_mask):
            return None, valid_mask

        expr_stack = torch.stack([e if isinstance(e, torch.Tensor) else torch.tensor(e) for e in expressions])
        sf = torch.tensor(self.size_factor_val).float()
        fmap = F.softmax(expr_stack, dim=-1) * sf

        distr = NegativeBinomial(mu=fmap, theta=torch.exp(self.vq_model.theta))
        counts = distr.sample_ori().detach().numpy()

        return counts, valid_mask

    # ---------- get ground-truth expression (cached) ----------
    def _get_gt_expression(self, cell_id, dataset):
        if cell_id in self._gt_expr_cache:
            return self._gt_expr_cache[cell_id]
        gene_names, expr_values, *_ = dataset.get_next_cell_velo(cell_id, dataset.global_dataset)
        expr = np.array(
            pd.Series(expr_values, index=gene_names)
            .reindex(self.reference_gene, fill_value=0).values,
            dtype=np.float64
        )
        self._gt_expr_cache[cell_id] = expr
        return expr

    # ---------- distributed generation (all ranks) ----------
    @torch.no_grad()
    def _generate_on_split(self, split_name, dataset, model, device, rank, world_size):
        """Run model generation on this rank's shard of the dataset."""
        raw_dataset = dataset
        if self.sample_size is not None and len(dataset) > self.sample_size:
            indices = random.sample(range(len(dataset)), self.sample_size)
            dataset = torch.utils.data.Subset(dataset, indices)

        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(
            dataset, batch_size=self.eval_batch_size,
            collate_fn=collate_fn_infer_perturb_vq, shuffle=False,
            sampler=sampler,
        )

        local_gen_expr = []
        local_gt_pert = []
        local_gt_ctrl = []
        local_cell_lines = []
        local_perturbations = []
        history_ids = {}

        desc = f"[PerturbEval] {split_name} (rank {rank})" if rank == 0 else None
        iterator = tqdm(dataloader, desc=desc) if rank == 0 else dataloader

        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            cell_pos = batch['cell_pos'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            drug_emb = batch['drug_emb'].to(device) if batch.get('drug_emb') is not None else None
            prefix_len = input_ids.shape[1]

            generated_ids, _, _ = model.generate_debug(
                input_ids=input_ids,
                cell_pos=cell_pos,
                max_new_tokens=self.max_new_tokens,
                ignore_Idx=self.ignore_ids,
                top_k=1,
                use_cache=True,
                debug=False,
                attention_mask=attn_mask,
                drug_emb=drug_emb,
            )

            gen_ids_list = generated_ids.cpu().tolist()
            counts, valid_mask = self._decode_vq_to_expr(gen_ids_list, prefix_len)
            if counts is None:
                continue

            for i, idx in enumerate(batch['idx']):
                if not valid_mask[i]:
                    continue
                pk = raw_dataset.perturb_key[idx]
                pair = raw_dataset.pairs[pk]

                local_gen_expr.append(counts[i])
                local_cell_lines.append(pk[1])
                local_perturbations.append(pk[2])

                if pk not in history_ids:
                    history_ids[pk] = 0
                hi = history_ids[pk]

                pert_id = pair['pert_ids'][hi % len(pair['pert_ids'])]
                local_gt_pert.append(self._get_gt_expression(pert_id, raw_dataset))

                ctrl_id = pair['ctrl_ids'][hi % len(pair['ctrl_ids'])]
                local_gt_ctrl.append(self._get_gt_expression(ctrl_id, raw_dataset))

                history_ids[pk] = hi + 1

        return local_gen_expr, local_gt_pert, local_gt_ctrl, local_cell_lines, local_perturbations

    # ---------- gather results from all ranks to rank 0 ----------
    def _gather_results(self, local_gen_expr, local_gt_pert, local_gt_ctrl,
                        local_cell_lines, local_perturbations, device):
        """Gather variable-length results from all ranks onto rank 0."""
        n_genes = len(self.reference_gene)
        local_n = len(local_gen_expr)

        if local_n > 0:
            gen_t = torch.tensor(np.stack(local_gen_expr), dtype=torch.float32, device=device)
            pert_t = torch.tensor(np.stack(local_gt_pert), dtype=torch.float64, device=device)
            ctrl_t = torch.tensor(np.stack(local_gt_ctrl), dtype=torch.float64, device=device)
        else:
            gen_t = torch.zeros(0, n_genes, dtype=torch.float32, device=device)
            pert_t = torch.zeros(0, n_genes, dtype=torch.float64, device=device)
            ctrl_t = torch.zeros(0, n_genes, dtype=torch.float64, device=device)

        local_count = torch.tensor([local_n], dtype=torch.long, device=device)
        world_size = dist.get_world_size()

        all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        counts_list = [c.item() for c in all_counts]
        max_n = max(counts_list) if max(counts_list) > 0 else 1

        def _pad_and_gather(tensor, max_n, n_genes, dtype):
            padded = torch.zeros(max_n, n_genes, dtype=dtype, device=device)
            if tensor.shape[0] > 0:
                padded[:tensor.shape[0]] = tensor
            gathered = [torch.zeros(max_n, n_genes, dtype=dtype, device=device) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            parts = [gathered[r][:counts_list[r]] for r in range(world_size)]
            return torch.cat(parts, dim=0).cpu().numpy()

        all_gen = _pad_and_gather(gen_t, max_n, n_genes, torch.float32)
        all_pert = _pad_and_gather(pert_t, max_n, n_genes, torch.float64)
        all_ctrl = _pad_and_gather(ctrl_t, max_n, n_genes, torch.float64)

        # gather string metadata via pickle through all_gather_object
        all_cl_lists = [None] * world_size
        all_pt_lists = [None] * world_size
        dist.all_gather_object(all_cl_lists, local_cell_lines)
        dist.all_gather_object(all_pt_lists, local_perturbations)

        merged_cl = []
        merged_pt = []
        for r in range(world_size):
            merged_cl.extend(all_cl_lists[r])
            merged_pt.extend(all_pt_lists[r])

        return all_gen, all_pert, all_ctrl, merged_cl, merged_pt

    # ---------- compute metrics (rank 0 only) ----------
    def _compute_metrics(self, split_name, gen_expr, gt_pert, gt_ctrl, cell_lines, perturbations):
        n = gen_expr.shape[0]
        if n == 0:
            return {}

        adata_pred = ad.AnnData(np.concatenate([gen_expr, gt_ctrl]))
        adata_pred.obs['celltype'] = cell_lines + cell_lines
        adata_pred.obs['perturbation'] = perturbations + ['control'] * n
        sc.pp.normalize_total(adata_pred, target_sum=1e4)
        sc.pp.log1p(adata_pred)

        adata_real = ad.AnnData(np.concatenate([gt_pert, gt_ctrl]))
        adata_real.obs['celltype'] = cell_lines + cell_lines
        adata_real.obs['perturbation'] = perturbations + ['control'] * n
        sc.pp.normalize_total(adata_real, target_sum=1e4)
        sc.pp.log1p(adata_real)

        from cell_eval import MetricsEvaluator
        from cell_eval._pipeline import MetricPipeline
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert='control',
            pert_col='perturbation',
            num_threads=32,
        )
        pipeline = MetricPipeline(profile=None)
        pipeline.add_metrics(self.METRICS_TO_LOG)
        pipeline.compute_de_metrics(evaluator.de_comparison)
        pipeline.compute_anndata_metrics(evaluator.anndata_pair)
        agg_results = pipeline.get_agg_results()
        mean_row = agg_results.filter(pl.col('statistic') == 'mean').to_dicts()
        if not mean_row:
            return {}
        mean_dict = mean_row[0]

        metrics = {}
        for m in self.METRICS_TO_LOG:
            if m in mean_dict:
                metrics[f"eval_{split_name}/{m}"] = mean_dict[m]
        return metrics

    # ---------- callbacks ----------
    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_before_train and not self._did_eval_before_train:
            self._did_eval_before_train = True
            self._run_eval(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_step_per != 0:
            return
        self._run_eval(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._run_eval(state)

    def _run_eval(self, state):
        trainer = self.trainer
        local_rank = trainer.args.local_rank
        is_main = is_main_process(local_rank=local_rank)
        is_distributed = dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        eval_seed = 42 + state.global_step
        random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)

        model = trainer.model
        unwrapped = model.module if hasattr(model, 'module') else model
        was_training = unwrapped.training
        unwrapped.eval()
        device = next(unwrapped.parameters()).device

        log_data = {"global_step": state.global_step}

        for split_name, dataset in self.eval_datasets_gen.items():
            try:
                local_gen, local_pert, local_ctrl, local_cl, local_pt = \
                    self._generate_on_split(split_name, dataset, unwrapped, device, rank, world_size)

                if is_distributed and world_size > 1:
                    gen_expr, gt_pert, gt_ctrl, cell_lines, perturbations = \
                        self._gather_results(local_gen, local_pert, local_ctrl, local_cl, local_pt, device)
                else:
                    gen_expr = np.stack(local_gen) if local_gen else np.zeros((0, len(self.reference_gene)))
                    gt_pert = np.stack(local_pert) if local_pert else np.zeros((0, len(self.reference_gene)))
                    gt_ctrl = np.stack(local_ctrl) if local_ctrl else np.zeros((0, len(self.reference_gene)))
                    cell_lines, perturbations = local_cl, local_pt

                if is_main:
                    metrics = self._compute_metrics(split_name, gen_expr, gt_pert, gt_ctrl, cell_lines, perturbations)
                    log_data.update(metrics)
                    print(f"[PerturbEval] step={state.global_step} {split_name}: {metrics}")
            except Exception as e:
                if is_main:
                    print(f"[PerturbEval] step={state.global_step} {split_name} failed: {e}")
                    import traceback
                    traceback.print_exc()

        if is_main and log_data and len(log_data) > 1:
            wandb.log(log_data, step=state.global_step, commit=False)

        if was_training:
            unwrapped.train()

        if is_distributed:
            dist.barrier()
