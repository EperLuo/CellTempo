import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as ADataset
from datasets import load_dataset,load_from_disk, concatenate_datasets, DatasetDict
import os, glob
from torch.utils.data import Dataset
import numpy as np
from utils.tokenizer import mixMulanTokenizer
import json
from loguru import logger
from typing import List
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import random
import pandas as pd
from torch.utils.data import Sampler
import scipy.sparse as sp
import pyarrow.parquet as pq
import scanpy as sc
import anndata as ad
import gzip
import pickle
import sys
from pathlib import Path
vq_path = os.path.abspath('..')
sys.path.append(vq_path)
BASE_DIR = Path(__file__).resolve().parent
from model.CellTempo_VQVAE.model import VQModel


def map_adata_to_reference_genes(adata, ref_genes):
    """
    Map genes in adata to the reference gene list ref_genes.
    - Missing genes are zero-padded.
    - Extra genes are dropped.
    - The output adata.var follows the order of ref_genes.

    Args:
        adata: AnnData object.
        ref_genes: list[str], reference gene list (target gene order).
    Returns:
        A new AnnData object.
    """
    # existing gene names
    current_genes = np.array(adata.var_names)

    # find intersection and index mapping
    intersect_genes = np.intersect1d(current_genes, ref_genes)
    missing_genes = [g for g in ref_genes if g not in current_genes]

    print(f"✅ {len(intersect_genes)} genes matched, "
          f"{len(missing_genes)} missing from adata.")

    # extract expression matrix for intersecting genes
    adata_aligned = adata[:, intersect_genes].copy()

    # zero-pad missing genes
    if missing_genes:
        import scipy.sparse as sp
        n_cells = adata_aligned.n_obs
        zero_mat = sp.csr_matrix((n_cells, len(missing_genes)))
        from anndata import AnnData
        adata_missing = ad.AnnData(X=zero_mat)
        adata_missing.var_names = missing_genes
        adata_missing.obs_names = adata_aligned.obs_names
        adata_aligned = ad.concat([adata_aligned, adata_missing], axis=1)

    # reorder to match ref_genes order
    adata_aligned = adata_aligned[:, ref_genes].copy()
    adata_aligned.obs = adata.obs

    return adata_aligned


def collate_fn(batch):
    # extract tokens and values
    tokens = [torch.tensor(item['tokens']) for item in batch]
    values = [torch.tensor(item['values']) for item in batch]
    data_len = torch.tensor([torch.tensor(item['trunc_full_len']) for item in batch])
    c1_len = torch.tensor([torch.tensor(item['c1_len']) for item in batch])
    c2_start = torch.tensor([torch.tensor(item['c2_start']) for item in batch])
    
    
    # generate labels by shifting the sequence left by one position
    y_t = [torch.cat([t[1:], torch.tensor([0])]) for t in tokens]  # next-token labels
    y_v = [torch.cat([v[1:], torch.tensor([0])]) for v in values]  # same for values

    # pad sequences
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    values_padded = pad_sequence(values, batch_first=True, padding_value=0)
    y_t_padded = pad_sequence(y_t, batch_first=True, padding_value=0)
    y_v_padded = pad_sequence(y_v, batch_first=True, padding_value=0)
    
    # build cell_pos tensor (same shape as tokens_padded) using c1_len
    cell_pos = torch.zeros_like(tokens_padded)
    for i, c1_ln in enumerate(c1_len):
        c2_st = c2_start[i]
        cell_pos[i, c1_ln:c2_st] = 1
        cell_pos[i, c2_st:] = 2

    # return processed batch
    return {
        'tokens': tokens_padded,
        'values': values_padded,
        'labels_tokens': y_t_padded,
        'labels_values': y_v_padded,
        'data_len': data_len,
        'c1_len': c1_len,
        'c2_start': c2_start,
        'cell_pos': cell_pos,
    }

def collate_fn_train_traj_vq(batch):

    tokens = [item['tokens'] for item in batch]
    cell_pos_list = [item['cell_pos'] for item in batch]

    data_len = torch.tensor([item['trunc_full_len'] for item in batch])
    c1_len = torch.tensor([item['c1_len'] for item in batch])
    c2_start = torch.tensor([item['c2_start'] for item in batch])
    
    # generate labels by shifting the sequence left by one position
    y_t = [torch.cat([torch.tensor(t[1:]), torch.tensor([0])]) for t in tokens]

    # pad sequences
    end_token = tokens[0][-1]
    tokens_padded = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=end_token)
    y_t_padded = pad_sequence(y_t, batch_first=True, padding_value=end_token)

    # padding value is 0
    cell_pos = pad_sequence([torch.tensor(c) for c in cell_pos_list], batch_first=True, padding_value=0)

    # return processed batch
    return {
        'input_ids': tokens_padded, 
        'labels': y_t_padded,
        'xlen': data_len,
        'c1_len': c1_len,
        'c2_start': c2_start,
        'cell_pos': cell_pos,
    }

def collate_fn_infer_traj_vq(batch):
    """
    Collate function that supports different target_id values within a batch.
    Uses left-padding so that the rightmost (most recent) tokens are aligned,
    which is required for correct autoregressive generation with KV cache.
    """
    if not batch:
        return None

    c2_start_values = [item['c2_start'] for item in batch]
    unique_c2_start = set(c2_start_values)
    assert len(unique_c2_start) == 1, f"Batch samples have different c2_start values: {unique_c2_start}"

    c2_start_val = c2_start_values[0]

    # Each sample truncated by its OWN target_id
    token_lists = []
    cell_pos_lists = []
    target_ids = []
    for item in batch:
        tid = item.get('target_id', 1)
        target_ids.append(tid)
        length = c2_start_val * tid + 2
        token_lists.append(torch.tensor(item['tokens'][:length]))
        cell_pos_lists.append(torch.tensor(item['cell_pos'][:length]))

    max_len = max(t.size(0) for t in token_lists)

    # Left-pad to max_len
    padded_tokens = []
    padded_cell_pos = []
    attention_masks = []
    pad_lens = []
    for tokens, pos in zip(token_lists, cell_pos_lists):
        pad_len = max_len - tokens.size(0)
        pad_lens.append(pad_len)
        if pad_len > 0:
            padded_tokens.append(torch.cat([torch.zeros(pad_len, dtype=tokens.dtype), tokens]))
            padded_cell_pos.append(torch.cat([torch.zeros(pad_len, dtype=pos.dtype), pos]))
            attention_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.bool),
                torch.ones(tokens.size(0), dtype=torch.bool),
            ]))
        else:
            padded_tokens.append(tokens)
            padded_cell_pos.append(pos)
            attention_masks.append(torch.ones(tokens.size(0), dtype=torch.bool))

    input_ids = torch.stack(padded_tokens)
    cell_pos = torch.stack(padded_cell_pos)
    attention_mask = torch.stack(attention_masks)

    c1_len = torch.tensor([item['c1_len'] for item in batch], dtype=torch.long)
    c2_start = torch.tensor(c2_start_values, dtype=torch.long)
    idx = [item['idx'] for item in batch]

    token_labels = pad_sequence([
        torch.tensor(item['tokens'][c2_start_val * item.get('target_id', 1) + 2:])
        for item in batch
    ], batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'c1_len': c1_len,
        'c2_start': c2_start,
        'cell_pos': cell_pos,
        'attention_mask': attention_mask,
        'token_labels': token_labels,
        'idx': idx,
        'pad_lens': torch.tensor(pad_lens, dtype=torch.long),
    }


def collate_fn_train_target_vq_perturb(batch):

    # 分别提取tokens和values
    tokens = [item['tokens'] for item in batch]
    # values = [item['values'] for item in batch]
    cell_pos_list = [item['cell_pos'] for item in batch]


    data_len = torch.tensor([item['trunc_full_len'] for item in batch])
    c1_len = torch.tensor([item['c1_len'] for item in batch])
    c2_start = torch.tensor([item['c2_start'] for item in batch])
    
    # 生成标签，这里简单地将序列向左移动一位
    y_t = [torch.cat([torch.tensor(t[1:]), torch.tensor([-100])]) for t in tokens]

    # 填充或截断
    end_token = tokens[0][-1]
    tokens_padded = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=end_token)
    y_t_padded = pad_sequence(y_t, batch_first=True, padding_value=-100)
    # y_v_padded = pad_sequence(y_v, batch_first=True, padding_value=0)

    # padding值设为2
    cell_pos = pad_sequence([torch.tensor(c) for c in cell_pos_list], batch_first=True, padding_value=0)

    # stack precomputed drug embeddings if available
    if 'drug_emb' in batch[0]:
        drug_emb = torch.stack([item['drug_emb'] for item in batch])  # (B, drug_emb_dim)
    else:
        drug_emb = None

    return {
        'input_ids': tokens_padded,
        'labels': y_t_padded,
        'xlen': data_len,
        'c1_len': c1_len,
        'c2_start': c2_start,
        'cell_pos': cell_pos,
        'drug_emb': drug_emb,
    }



def collate_fn_infer_perturb_vq(batch):
    """
    Collate function for perturb inference.
    Truncates each sample to the prefix (up to 'perturb' + '<S>'),
    keeping the rest as ground-truth labels.
    Perturb sequence: [plate, cell_line, drug, dose, control, <S>, ...vq_codes..., <E>, perturb, <S>, ...]
    prefix_len = 4 (meta) + c1_len (control+<S>+codes+<E>) + 2 (perturb+<S>)
    """
    if not batch:
        return None

    META_TOKENS = 4  # plate, cell_line, drug, dose

    prefix_lens = []
    for item in batch:
        # c1_len = 1(<S>) + N(vq codes) + 1(<E>) + 1(control) = N+3
        # prefix = 3(meta) + c1_len + 2(perturb + <S>)
        plen = META_TOKENS + item['c1_len'] + 2
        prefix_lens.append(plen)

    # all samples should have the same prefix length
    assert len(set(prefix_lens)) == 1, f"Inconsistent prefix lengths: {set(prefix_lens)}"
    prefix_len = prefix_lens[0]

    input_ids = torch.stack([torch.tensor(item['tokens'][:prefix_len]) for item in batch])
    cell_pos = torch.stack([torch.tensor(item['cell_pos'][:prefix_len]) for item in batch])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    c1_len = torch.tensor([item['c1_len'] for item in batch], dtype=torch.long)
    c2_start = torch.tensor([item['c2_start'] for item in batch], dtype=torch.long)
    idx = [item['idx'] for item in batch]

    token_labels = pad_sequence([
        torch.tensor(item['tokens'][prefix_len:]) for item in batch
    ], batch_first=True, padding_value=-100)

    if 'drug_emb' in batch[0]:
        drug_emb = torch.stack([item['drug_emb'] for item in batch])
    else:
        drug_emb = None

    return {
        'input_ids': input_ids,
        'c1_len': c1_len,
        'c2_start': c2_start,
        'cell_pos': cell_pos,
        'attention_mask': attention_mask,
        'token_labels': token_labels,
        'drug_emb': drug_emb,
        'idx': idx,
    }


def load_and_concatenate_shards(parent_dir: str, expect_features=None):
    """Load shards and concatenate into a single Dataset (zero-copy merge)."""
    dirs = [d for d in glob.glob(os.path.join(parent_dir, "part_*")) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"No shards under {parent_dir}")
    # sort by shard index
    import re as _re
    dirs = sorted(dirs, key=lambda p: int(_re.search(r"part_(\d+)", p).group(1)))

    parts = [load_from_disk(p) for p in tqdm(dirs, desc="Loading datasets")]

    return concatenate_datasets(parts)

def filter_by_names(dataset, dataset_names, train_flag=True, num_proc=8, batch_size=100_000):
    """
    dataset: Dataset or DatasetDict (if DatasetDict, each split is filtered separately).
    dataset_names: list of dataset names to reserve as the test set.
    train_flag: True selects the training set (not in list), False selects the test set (in list).
    num_proc: number of parallel processes.
    batch_size: batch size for batched filtering (adjust based on available memory).
    """
    names = set(dataset_names)

    def keep_batch(batch):
        # vectorized boolean check
        if train_flag:
            return [nm.split('/')[-1] not in names for nm in batch]
        else:
            return [nm.split('/')[-1] in names for nm in batch]

    if isinstance(dataset, DatasetDict):
        # filter each split separately
        return DatasetDict({
            split: ds.filter(
                keep_batch, input_columns=["dataset_name"],
                batched=True, batch_size=batch_size,
                num_proc=num_proc, desc=f"Filter {split} ({'train' if train_flag else 'test'})"
            )
            for split, ds in dataset.items()
        })
    else:
        # single Dataset
        return dataset.filter(
            keep_batch, input_columns=["dataset_name"],
            batched=True, batch_size=batch_size,
            num_proc=num_proc, desc=f"Filter ({'train' if train_flag else 'test'})"
        )

class scBasetraj_vq(Dataset): # Accepts multiple HuggingFace datasets
    ## Currently compatible: Velocity ✅, scperturb ✅
    ## For the second cell, only the Top 100 genes are used instead of the full gene set.
    ## Resolved: how to obtain the Top 100 gene list. ✅
    ## FIXME: currently only compatible with perturbation tasks, not velocity; pre-training can come first.
    ## FIXME: add positional encoding to inter-cell tokens so order is preserved between cells.

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # only needed for velocity; used to find next cell globally
                 mode: str = 'train',
                 global_dataset: str = 'velo_dataset_all', # only needed for velocity; used to find next cell globally
                 data_types: list = ['trajectory','perturb'], # supported types: sc-rna, velocity, perturb
                 dataset: ADataset = None,
                 vq_vae_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel'
                ):

        velo_data_indx = data_types.index('trajectory')
        self.global_dataset = dataset #load_from_disk(os.path.join(data_folders[velo_data_indx],global_dataset))['train']
        with open(os.path.join(data_folders[velo_data_indx], mapping_dict), 'r') as f:
            self.cell_name_to_num = json.load(f)
            self.all_cell_name = self.cell_name_to_num.keys()

        with open(os.path.join(data_folders[0], meta_info_name), 'r') as f:
            self.meta_info = json.load(f)

        self.__chars = self.meta_info['token_set']
        self.vocab_size = len(self.__chars)
        self.tokenizer = mixMulanTokenizer(self.__chars)
        self.crop_train_length = crop_train_length
        self.data_types = data_types

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.success('Loading data from {} succeed'.format(data_folders))
            logger.info(f'vocab size is {self.vocab_size}')
            logger.info(f'cropped data_block_size  is {crop_train_length}')
        
        self.reference_gene = pd.read_csv(str(BASE_DIR / 'OS_scRNA_gene_index.18791.tsv'), sep='\t')['gene_name'].values
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index of the current process
        device = torch.device(f"cuda:{local_rank}")
        print('current device: ', device)
        self.vq_model = VQModel.from_pretrained(vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')#.to(device=device)

        table = pq.read_table(os.path.join(data_folders[0], f'trajectory_{mode}.parquet'))
        self.trajectory_list = table['tokens']#.to_pylist()

            
    def __len__(self):            
        return len(self.trajectory_list)
        
    def __getitem__(self, idx_num):

        data_type = self.data_types[0]
        traj = self.trajectory_list[idx_num].as_py()[::3]
        while traj[0] not in self.all_cell_name:  # some cells were dropped during preprocessing
            idx_num += 1
            traj = self.trajectory_list[idx_num].as_py()[::3]
        
        processed_cell = self.extract_gene_and_expr(traj)
        paired_cell = self.concat_and_trunc_cell(processed_cell, trunc = self.crop_train_length) # if autoregressive
        paired_cell['idx'] = idx_num

        return paired_cell
    
    def get_next_cell_velo(self, idx_num, global_dataset):
        if idx_num is None:
            return None, None
        cell_data = global_dataset[idx_num]
        gene_names = cell_data['expressed_genes']
        expr_values = cell_data['expressed_values']
        return gene_names, expr_values
    
    def extract_gene_and_expr(self, idx_num):
        expr_values = []
        for next_cell_name in idx_num:
            try:
                next_cell_global_id = self.cell_name_to_num[next_cell_name]
            except:
                continue
            gene_names_next_cell, expr_values_next_cell = self.get_next_cell_velo(next_cell_global_id, self.global_dataset)
            expr_values_next_cell = torch.tensor(pd.Series(expr_values_next_cell, index=gene_names_next_cell).reindex(self.reference_gene, fill_value=0).values, dtype=torch.float32)
            expr_values.append(expr_values_next_cell)
        instruction = None

        expr_vec = torch.stack(expr_values)
        latent = self.vq_model.encode(expr_vec).latents
        quant, _, (perplexity, min_encodings, encoding_indices) = self.vq_model.quantize(latent)

        processed_cell = {
            'values': encoding_indices.reshape(expr_vec.shape[0],-1).cpu().numpy(),
            'instruction':instruction,
        }

        return processed_cell
    
    def concat_and_trunc_cell(self, processed_cell, trunc = True):
        
        start_tokens = ['<S>']

        # store instruction info
        instruction = {}
        end_tokens = ['<E>']
        token_cell = processed_cell['values'].astype(str) #[str(index) for index in processed_cell['values']]

        tokens = []
        cell_pos = []   # 0 pad, 1 meta info, >=2 cell id 
        for i, token in enumerate(token_cell):
            tokens += [f"traj_{i}"] + start_tokens + list(token) + end_tokens
            cell_pos += [1]
            cell_pos += [i+2] * (len(start_tokens) + token.shape[0] + len(end_tokens))

        c1_len = len(start_tokens) + token.shape[0] + len(end_tokens) + 1
        
        token_ids = self.tokenizer.encode(tokens)
        full_length = len(token_ids)
        
        concated_cell = {
            'tokens': token_ids,  
            'c1_len': c1_len, # without inter tokens
            'c2_start': c1_len, # including inter tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # save instruction info
        }
        return concated_cell

class h5ad_data_vq(Dataset): # Accepts multiple HuggingFace datasets
    ## Currently compatible: Velocity ✅, scperturb ✅
    ## For the second cell, only the Top 100 genes are used instead of the full gene set.
    ## Resolved: how to obtain the Top 100 gene list. ✅
    ## FIXME: currently only compatible with perturbation tasks, not velocity; pre-training can come first.
    ## FIXME: add positional encoding to inter-cell tokens so order is preserved between cells.

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # only needed for velocity; used to find next cell globally
                 mode: str = 'train',
                 data_types: list = ['trajectory','perturb'], # supported types: sc-rna, velocity, perturb
                 global_dataset: str = 'velo_dataset_all', # only needed for velocity; used to find next cell globally
                 vq_vae_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel'
                ):

        velo_data_indx = data_types.index('trajectory')
        self.global_dataset = sc.read_h5ad(os.path.join(data_folders[velo_data_indx],dataset_names[velo_data_indx]))
        self.global_dataset.var_names = self.global_dataset.var_names.str.upper()
        self.global_dataset = self.global_dataset[:, ~self.global_dataset.var_names.duplicated()].copy()
        # with open(os.path.join(data_folders[velo_data_indx], mapping_dict), 'r') as f:
        #     self.cell_name_to_num = json.load(f)
        #     self.all_cell_name = self.cell_name_to_num.keys()

        with open(os.path.join(data_folders[0], meta_info_name), 'r') as f:
            self.meta_info = json.load(f)

        self.__chars = self.meta_info['token_set']
        self.vocab_size = len(self.__chars)
        self.tokenizer = mixMulanTokenizer(self.__chars)
        self.crop_train_length = crop_train_length
        
        
        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.success('Loading data from {} succeed'.format(data_folders))
            logger.info(f'vocab size is {self.vocab_size}')
            logger.info(f'cropped data_block_size  is {crop_train_length}')
        
        self.reference_gene = pd.read_csv(str(BASE_DIR / 'OS_scRNA_gene_index.18791.tsv'), sep='\t')['gene_name'].values
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index of the current process
        device = torch.device(f"cuda:{local_rank}")
        print('current device: ', device)
        self.vq_model = VQModel.from_pretrained(vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')#.to(device=device)

        self.global_dataset = map_adata_to_reference_genes(self.global_dataset, self.reference_gene)
        self.global_dataset = self.global_dataset.X.toarray() if sp.issparse(self.global_dataset.X) else self.global_dataset.X
            
    def __len__(self):            
        return self.global_dataset.shape[0]
        
    def __getitem__(self, idx_num):
        processed_cell = self.extract_gene_and_expr(idx_num)
        paired_cell = self.concat_and_trunc_cell(processed_cell, trunc = self.crop_train_length) # if autoregressive
        paired_cell['idx'] = idx_num

        return paired_cell
    
    def get_next_cell_velo(self, idx_num, global_dataset):

        if idx_num is None:
            # print(f'idx num {idx_num} is None.')
            return None, None
        cell_data = global_dataset[idx_num]
        gene_names = cell_data['expressed_genes']
        expr_values = cell_data['expressed_values']

        return gene_names, expr_values
   
    def extract_gene_and_expr(self, idx_num):

        expr_values = [] 
        expr_values.append(torch.tensor(self.global_dataset[idx_num],dtype=torch.float32))
        instruction = None

        expr_vec = torch.stack(expr_values)
        latent = self.vq_model.encode(expr_vec).latents
        quant, _, (perplexity, min_encodings, encoding_indices) = self.vq_model.quantize(latent)

        processed_cell = {
            'values': encoding_indices.reshape(expr_vec.shape[0],-1).cpu().numpy(),
            'instruction':instruction,
        }

        return processed_cell
    
    def concat_and_trunc_cell(self, processed_cell, trunc = True):
        
        start_tokens = ['<S>']

        # store instruction info
        instruction = {}
        end_tokens = ['<E>']
        token_cell = processed_cell['values'].astype(str) #[str(index) for index in processed_cell['values']]

        tokens = []
        cell_pos = []
        for i, token in enumerate(token_cell):
            tokens += [f"traj_{i}"] + start_tokens + list(token) + end_tokens
            cell_pos += [1]
            cell_pos += [i+2] * (len(start_tokens) + token.shape[0] + len(end_tokens))
        
        tokens += [f"traj_1"] + start_tokens
        cell_pos += [1]
        cell_pos += [3] * (len(start_tokens))

        c1_len = len(start_tokens) + token.shape[0] + len(end_tokens) + 1
        
        token_ids = self.tokenizer.encode(tokens)
        full_length = len(token_ids)
        
        concated_cell = {
            'tokens': token_ids,  
            'c1_len': c1_len, # without inter tokens
            'c2_start': c1_len, # including inter tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # save instruction info
        }
        return concated_cell


class Tahoe100m_vq(Dataset): # Accepts multiple HuggingFace datasets
    ## Currently compatible: Velocity ✅, scperturb ✅
    ## For the second cell, only the Top 100 genes are used instead of the full gene set.
    ## Resolved: how to obtain the Top 100 gene list. ✅
    ## FIXME: currently only compatible with perturbation tasks, not velocity; pre-training can come first.
    ## FIXME: add positional encoding to inter-cell tokens so order is preserved between cells.

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 n_express_level: int = 10,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # only needed for velocity; used to find next cell globally
                 mode: str = 'train',
                 global_dataset: str = 'velo_dataset_all', # only needed for velocity; used to find next cell globally
                 dataset: ADataset = None,
                 vq_vae_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel',
                 drug_emb_paths: list = [],
                 repeat_per_pair: int = 1,
                ):

        with open(os.path.join(data_folders[0], meta_info_name), 'r') as f:
            self.meta_info = json.load(f)

        self.__chars = self.meta_info['token_set']
        self.vocab_size = len(self.__chars)
        self.tokenizer = mixMulanTokenizer(self.__chars)
        self.crop_train_length = crop_train_length
        self.n_express_level = n_express_level
        
        self.global_dataset = dataset

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.success('Loading data from {} succeed'.format(data_folders))
            logger.info(f'vocab size is {self.vocab_size}')
            logger.info(f'cropped data_block_size  is {crop_train_length}')
        
        self.reference_gene = pd.read_csv(str(BASE_DIR / 'OS_scRNA_gene_index.18791.tsv'), sep='\t')['gene_name'].values
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index of the current process
        device = torch.device(f"cuda:{local_rank}")
        print('current device: ', device)
        self.vq_model = VQModel.from_pretrained(vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')#.to(device=device)

        # load precomputed drug molecular embeddings (UniMol)
        self.drug_emb_cache = {}
        for emb_path in drug_emb_paths:
            emb_dict = torch.load(emb_path, map_location='cpu')
            for smiles, emb in emb_dict.items():
                self.drug_emb_cache[smiles] = torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb.float()
        if self.drug_emb_cache and ('LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0'):
            logger.info(f'Loaded {len(self.drug_emb_cache)} drug embeddings, dim={next(iter(self.drug_emb_cache.values())).shape[0]}')

        pairs_path = os.path.join(data_folders[0],dataset_names[0],f'pairs_{mode}.json.gz')
        with gzip.open(pairs_path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        self.pairs = {}
        for k, v in obj.items():
            plate, cell_line, drug = k.split("||", 2)
            self.pairs[(plate, cell_line, drug)] = {
                "pert_ids": [int(x) for x in v["pert_ids"]],
                "ctrl_ids": [int(x) for x in v["ctrl_ids"]],
            }
        self.perturb_key = list(self.pairs.keys())

        gene_metadata_path = os.path.join(data_folders[0], dataset_names[0], "metadata", "gene_metadata.parquet")
        gene_metadata_df = pd.read_parquet(gene_metadata_path)
        self.gene_vocab = {}
        for _, entry in gene_metadata_df.iterrows():
            self.gene_vocab[entry["token_id"]] = entry["gene_symbol"]
        sorted_vocab_items = sorted(self.gene_vocab.items())
        token_ids, gene_names = zip(*sorted_vocab_items)
        self.token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

        # build sample -> dose token mapping
        import ast
        sample_metadata_path = os.path.join(data_folders[0], dataset_names[0], "metadata", "sample_metadata.parquet")
        sample_df = pd.read_parquet(sample_metadata_path)
        self.dose = {}
        for _, row in sample_df.iterrows():
            try:
                parsed = ast.literal_eval(str(row['drugname_drugconc']))
                dose_val = parsed[0][1]
            except Exception:
                dose_val = 0.0
            self.dose[row['sample']] = f"dose_{dose_val}"

        self.repeat_per_pair = repeat_per_pair

    def __len__(self):            
        return len(self.pairs) * self.repeat_per_pair
        
    def __getitem__(self, idx_num):
        pair_idx = idx_num // self.repeat_per_pair
        repeat_idx = idx_num % self.repeat_per_pair

        pair = self.pairs[self.perturb_key[pair_idx]]
        pert_id = random.choice(pair['pert_ids'])
        ctrl_id = pair['ctrl_ids'][repeat_idx % len(pair['ctrl_ids'])]
        traj = [ctrl_id, pert_id]
        
        processed_cell = self.extract_gene_and_expr(traj,)
        paired_cell = self.concat_and_trunc_cell(processed_cell, trunc = self.crop_train_length) # if autoregressive
        paired_cell['idx'] = pair_idx

        smiles = processed_cell['smile']
        if self.drug_emb_cache and smiles in self.drug_emb_cache:
            paired_cell['drug_emb'] = self.drug_emb_cache[smiles]
        
        return paired_cell
    
    def get_next_cell_velo(self, idx_num, global_dataset):

        cell_data = global_dataset[idx_num]
        gene_names = cell_data['genes']
        expr_values = cell_data['expressions']
        drug=cell_data['drug']
        smiles=cell_data['canonical_smiles']
        plate=cell_data['plate']
        cell_line_id=cell_data['cell_line_id']
        dose=self.dose.get(cell_data['sample'], 'dose_0.0')

        if expr_values[0] < 0: 
            gene_names = gene_names[1:]
            expr_values = expr_values[1:]
        
        gene_names = [self.gene_vocab[gene] for gene in gene_names]

        return gene_names, expr_values, drug, smiles, plate, cell_line_id, dose
  
    def extract_gene_and_expr(self, idx_num):
        expr_values = [] 

        for next_cell_name in idx_num:
            gene_names_next_cell, expr_values_next_cell, drug, smiles, plate, cell_line_id, dose = self.get_next_cell_velo(next_cell_name, self.global_dataset)
            expr_values_next_cell = torch.tensor(pd.Series(expr_values_next_cell, index=gene_names_next_cell).reindex(self.reference_gene, fill_value=0).values, dtype=torch.float32)
            expr_values.append(expr_values_next_cell)
        instruction = None

        expr_vec = torch.stack(expr_values)
        latent = self.vq_model.encode(expr_vec).latents
        quant, _, (perplexity, min_encodings, encoding_indices) = self.vq_model.quantize(latent)

        processed_cell = {
            'values': encoding_indices.reshape(expr_vec.shape[0],-1).cpu().numpy(),
            'instruction':instruction,
            'drug':drug,
            'smile':smiles,
            'plate':plate,
            'cell_line':cell_line_id,
            'dose': dose,
        }

        return processed_cell
    
    def concat_and_trunc_cell(self, processed_cell, trunc = True):
        
        start_tokens = ['<S>']

        # store instruction info
        instruction = processed_cell['smile']
        end_tokens = ['<E>']
        token_cell = processed_cell['values'].astype(str) #[str(index) for index in processed_cell['values']]
        dose = processed_cell['dose']

        tokens = [str(processed_cell['plate']), str(processed_cell['cell_line']), 'drug', dose]
        cell_pos = [1,1,1,1]
        # for i, token in enumerate(token_cell):
        tokens += ["control"] + start_tokens + list(token_cell[0]) + end_tokens
        cell_pos += [1]
        cell_pos += [2] * (len(start_tokens) + token_cell[0].shape[0] + len(end_tokens))
        tokens += ["perturb"] + start_tokens + list(token_cell[1]) + end_tokens
        cell_pos += [1]
        cell_pos += [3] * (len(start_tokens) + token_cell[0].shape[0] + len(end_tokens))

        c1_len = len(start_tokens) + token_cell[0].shape[0] + len(end_tokens) + 1
        
        token_ids = self.tokenizer.encode(tokens)
        full_length = len(token_ids)
        
        concated_cell = {
            'tokens': token_ids,  
            'c1_len': c1_len, # without inter tokens
            'c2_start': c1_len, # including inter tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # save instruction info
        }
        return concated_cell


class h5ad_traj_vq(Dataset): # Accepts multiple HuggingFace datasets

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # only needed for velocity; used to find next cell globally
                 mode: str = 'train',
                 global_dataset: str = 'velo_dataset_all', # only needed for velocity; used to find next cell globally
                 data_types: list = ['trajectory','perturb'], # supported types: sc-rna, velocity, perturb
                 dataset: ADataset = None,
                 vq_vae_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel',
                 perturb_config: str = None,   # path to a YAML file with gene_modules and amplify_rules
                 trajectory_pkl: str = None,   # path to the pkl file containing (trajectory_list, target_id)
                ):

        velo_data_indx = data_types.index('trajectory')
        self.global_dataset = sc.read_h5ad(os.path.join(data_folders[velo_data_indx],dataset_names[velo_data_indx]))
        self.global_dataset.var_names = self.global_dataset.var_names.str.upper()
        self.global_dataset = self.global_dataset[:, ~self.global_dataset.var_names.duplicated()].copy()
        # with open(os.path.join(data_folders[velo_data_indx], mapping_dict), 'r') as f:
        #     self.cell_name_to_num = json.load(f)
        #     self.all_cell_name = self.cell_name_to_num.keys()

        with open(os.path.join(data_folders[0], meta_info_name), 'r') as f:
            self.meta_info = json.load(f)

        self.__chars = self.meta_info['token_set']
        self.vocab_size = len(self.__chars)
        self.tokenizer = mixMulanTokenizer(self.__chars)
        self.crop_train_length = crop_train_length

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.success('Loading data from {} succeed'.format(data_folders))
            logger.info(f'vocab size is {self.vocab_size}')
            logger.info(f'cropped data_block_size  is {crop_train_length}')
        
        self.reference_gene = pd.read_csv(str(BASE_DIR / 'OS_scRNA_gene_index.18791.tsv'), sep='\t')['gene_name'].values
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index of the current process
        device = torch.device(f"cuda:{local_rank}")
        print('current device: ', device)
        self.vq_model = VQModel.from_pretrained(vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')#.to(device=device)

        self.global_dataset = map_adata_to_reference_genes(self.global_dataset, self.reference_gene)

        # Apply gene amplification rules from external config (if provided).
        if perturb_config is not None:
            import yaml as _yaml
            with open(perturb_config, 'r') as _f:
                _pcfg = _yaml.safe_load(_f)
            _gene_modules = _pcfg.get('gene_modules', {})
            _amplify_rules = _pcfg.get('amplify_rules', [])
            _cluster_key = _pcfg.get('cluster_key', 'clusters')

            self.global_dataset.X = (
                self.global_dataset.X.toarray()
                if sp.issparse(self.global_dataset.X)
                else self.global_dataset.X
            )
            for rule in _amplify_rules:
                cluster   = rule['cluster']
                direction = rule['direction']   # "up" or "down"
                gene_list = _gene_modules[rule['module']][direction]
                add       = rule.get('add', 1.0)
                mul       = rule.get('mul', 1.0)
                mask = self.global_dataset.obs[_cluster_key] == cluster
                self.global_dataset[mask].X = self.amplify_genes(
                    self.global_dataset[mask], gene_list, add=add, mul=mul
                )
            self.global_dataset = self.global_dataset.X
        else:
            self.global_dataset = (
                self.global_dataset.X.toarray()
                if sp.issparse(self.global_dataset.X)
                else self.global_dataset.X
            )

        if trajectory_pkl is None:
            raise ValueError(
                "trajectory_pkl must be provided (path to the .pkl file "
                "containing (trajectory_list, target_id))."
            )
        with open(trajectory_pkl, "rb") as f:
            self.trajectory_list, self.target_id = pickle.load(f)

            
    def __len__(self):            
        return len(self.trajectory_list)
        
    def __getitem__(self, idx_num):

        traj = self.trajectory_list[idx_num]
        processed_cell = self.extract_gene_and_expr(traj)
        paired_cell = self.concat_and_trunc_cell(processed_cell, trunc = self.crop_train_length) # if autoregressive
        paired_cell['idx'] = idx_num
        paired_cell['target_id'] = self.target_id[idx_num]

        return paired_cell
    
    def extract_gene_and_expr(self, idx_num):

        expr_values = [] 

        for next_cell_name in idx_num:
            expr_values_next_cell = self.global_dataset[next_cell_name]
            expr_values.append(torch.tensor(expr_values_next_cell))
        instruction = None

        expr_vec = torch.stack(expr_values).to(torch.float32)
        latent = self.vq_model.encode(expr_vec).latents
        quant, _, (perplexity, min_encodings, encoding_indices) = self.vq_model.quantize(latent)

        processed_cell = {
            'values': encoding_indices.reshape(expr_vec.shape[0],-1).cpu().numpy(),
            'instruction':instruction,
        }

        return processed_cell
    
    def concat_and_trunc_cell(self, processed_cell, trunc = True):
        
        start_tokens = ['<S>']

        # store instruction info
        instruction = {}
        end_tokens = ['<E>']
        token_cell = processed_cell['values'].astype(str) #[str(index) for index in processed_cell['values']]

        tokens = []
        cell_pos = []   # 0 pad, 1 meta info, >=2 cell id 
        for i, token in enumerate(token_cell):
            tokens += [f"traj_{i}"] + start_tokens + list(token) + end_tokens
            cell_pos += [1]
            cell_pos += [i+2] * (len(start_tokens) + token.shape[0] + len(end_tokens))
        tokens += [f"traj_{i}"] + start_tokens
        cell_pos += [1]
        cell_pos += [i+2] * len(start_tokens)

        c1_len = len(start_tokens) + token.shape[0] + len(end_tokens) + 1
        
        token_ids = self.tokenizer.encode(tokens)
        full_length = len(token_ids)
        
        concated_cell = {
            'tokens': token_ids,  
            'c1_len': c1_len, # without inter tokens
            'c2_start': c1_len, # including inter tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # save instruction info
        }
        return concated_cell

    def amplify_genes(self, adata, gene_list, add=1.0, mul=5.0):
        """Amplify expression of specified genes: first add `add`, then multiply by `mul`."""
        # gene name → index
        gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

        # find gene indices (skip genes not present)
        idx = [gene_to_idx[g] for g in gene_list if g in gene_to_idx]
        if not idx:
            print("⚠️ No matching genes found!")
            return adata.X

        X = adata.X

        # sparse matrix handling
        if sp.issparse(X):
            # X[:, idx] += add
            for j in idx:
                col = X[:, j].toarray().flatten()  # safe copy
                col = (col + add) * mul
                X[:, j] = col.reshape(-1, 1)
            adata.X = X

        else:  # dense matrix
            adata.X[:, idx] = (adata.X[:, idx] + add) * mul

        return adata.X


class H5adPerturb_vq(Dataset):
    """Dataset for h5ad + single SMILES perturbation inference.

    Reads all cells from an h5ad file, pairs each cell with a given drug
    SMILES, and produces the same token format as Tahoe100m_vq so that the
    model generates perturbed cells.

    Drug embedding lookup order:
      1. Pre-computed cache from drug_emb_paths (same .pt format as Tahoe100m)
      2. On-the-fly computation via unimol_tools UniMolRepr (310M model)
    """

    def __init__(
        self,
        h5ad_path: str,
        smiles: str,
        data_folders: list = ['path1'],
        dataset_names: list = ['name1'],
        crop_train_length: int = 286,
        meta_info_name: str = 'mix_meta_info_vq_traj.json',
        vq_vae_path: str = '',
        drug_emb_paths: list = [],
        drug_emb_dim: int = 1024,
        plate: str = 'unknown_plate',
        cell_line: str = 'unknown_cell_line',
        dose: str = 'dose_0.0',
    ):
        with open(os.path.join(data_folders[0], meta_info_name), 'r') as f:
            self.meta_info = json.load(f)

        self.__chars = self.meta_info['token_set']
        self.vocab_size = len(self.__chars)
        self.tokenizer = mixMulanTokenizer(self.__chars)
        self.crop_train_length = crop_train_length
        self.smiles = smiles
        self.plate = plate
        self.cell_line = cell_line
        self.dose = dose

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.info(f'[H5adPerturb_vq] Loading h5ad from {h5ad_path}')

        self.reference_gene = pd.read_csv(
            str(BASE_DIR / 'OS_scRNA_gene_index.18791.tsv'), sep='\t'
        )['gene_name'].values

        self.vq_model = VQModel.from_pretrained(
            vq_vae_path, cvq_distance='cos', cvq_anchor='probrandom'
        )

        adata = sc.read_h5ad(h5ad_path)
        adata.var_names = adata.var_names.str.upper()
        adata = adata[:, ~adata.var_names.duplicated()].copy()
        adata = map_adata_to_reference_genes(adata, self.reference_gene)
        self.expr_matrix = (
            adata.X.toarray() if sp.issparse(adata.X) else adata.X
        )
        self.obs = adata.obs

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.info(f'[H5adPerturb_vq] Loaded {self.expr_matrix.shape[0]} cells')

        # resolve drug embedding
        self.drug_emb = self._resolve_drug_emb(
            smiles, drug_emb_paths, drug_emb_dim
        )

    def _resolve_drug_emb(self, smiles, drug_emb_paths, drug_emb_dim):
        """Look up embedding from cache; fall back to UniMolRepr computation."""
        for emb_path in drug_emb_paths:
            if not os.path.exists(emb_path):
                continue
            emb_dict = torch.load(emb_path, map_location='cpu')
            if smiles in emb_dict:
                emb = emb_dict[smiles]
                emb = torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb.float()
                logger.info(f'[H5adPerturb_vq] Found drug embedding in cache: {emb_path}')
                return emb

        logger.info(f'[H5adPerturb_vq] Drug embedding not found in cache, computing with UniMolRepr (310M)...')
        try:
            from unimol_tools import UniMolRepr
        except ImportError:
            raise ImportError(
                'unimol_tools is not installed. Install via: pip install unimol_tools huggingface_hub'
            )

        repr_model = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv2',
            model_size='310m',
            batch_size=1,
            use_cuda=torch.cuda.is_available(),
        )
        repr_output = repr_model.get_repr([smiles], return_atomic_reprs=False, return_tensor=True)
        emb = repr_output[0].float()
        assert emb.shape[0] == drug_emb_dim, (
            f'Expected drug_emb_dim={drug_emb_dim}, got {emb.shape[0]}'
        )
        logger.info(f'[H5adPerturb_vq] Computed embedding dim={emb.shape[0]} for SMILES: {smiles}')
        return emb

    def __len__(self):
        return self.expr_matrix.shape[0]

    def __getitem__(self, idx_num):
        expr_vec = torch.tensor(
            self.expr_matrix[idx_num], dtype=torch.float32
        ).unsqueeze(0)

        latent = self.vq_model.encode(expr_vec).latents
        _, _, (_, _, encoding_indices) = self.vq_model.quantize(latent)
        vq_codes = encoding_indices.reshape(1, -1).cpu().numpy()

        processed_cell = {
            'values': vq_codes,
            'smile': self.smiles,
            'plate': self.plate,
            'cell_line': self.cell_line,
            'dose': self.dose,
        }
        paired_cell = self._concat_and_trunc_cell(processed_cell)
        paired_cell['idx'] = idx_num
        paired_cell['drug_emb'] = self.drug_emb

        return paired_cell

    def _concat_and_trunc_cell(self, processed_cell):
        """Build token sequence: [plate, cell_line, drug, dose, control, <S>, ...codes..., <E>, perturb, <S>]"""
        start_tokens = ['<S>']
        end_tokens = ['<E>']
        token_cell = processed_cell['values'].astype(str)

        tokens = [str(processed_cell['plate']), str(processed_cell['cell_line']), 'drug', str(processed_cell['dose'])]
        cell_pos = [1, 1, 1, 1]

        tokens += ["control"] + start_tokens + list(token_cell[0]) + end_tokens
        cell_pos += [1]
        cell_pos += [2] * (len(start_tokens) + token_cell[0].shape[0] + len(end_tokens))

        tokens += ["perturb"] + start_tokens + list(token_cell[0]) + end_tokens
        cell_pos += [1]
        cell_pos += [3] * (len(start_tokens) + token_cell[0].shape[0] + len(end_tokens))

        c1_len = len(start_tokens) + token_cell[0].shape[0] + len(end_tokens) + 1

        token_ids = self.tokenizer.encode(tokens)
        full_length = len(token_ids)

        return {
            'tokens': token_ids,
            'c1_len': c1_len,
            'c2_start': c1_len,
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': self.smiles,
        }