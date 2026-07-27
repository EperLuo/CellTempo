#!/usr/bin/env python
"""
Drug screening pipeline for iPSC perturbation → trajectory → OPC differentiation analysis.

Workflow per drug molecule (default mode):
  1. generate_traj.py --infer_type perturb_h5ad  (predict perturbed cells)
  2. Decode VQ tokens, filter GPC_prolif, save merged h5ad & trajectory pkl
  3. generate_traj.py --infer_type trajectory_perturb_h5ad  (predict trajectory)
  4. Decode trajectory VQ tokens, marker-gene scoring, count OPC cells per step

DEG-trajectory mode (--deg_traj_mode):
  1. DMSO goes through the normal pipeline (perturb → trajectory)
  2. For each other drug:
     a. Perturbation generation (same as default)
     b. Compute DEGs: drug_perturbed vs DMSO_perturbed (Wilcoxon)
     c. Take top-N up/down DEGs → write perturb_gene config YAML
     d. Generate trajectory from DMSO's h5ad with per-drug gene amplification
     e. Decode trajectory, count cell types

Loops over all molecules listed in target_mole.txt and outputs a summary table
of OPC cell counts at each trajectory step for every molecule.
"""

import os
import sys
import json
import yaml
import pickle
import subprocess
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CELLTEMPO_DIR = os.path.dirname(SRC_DIR)
PROJECT_DIR = os.path.dirname(CELLTEMPO_DIR)
sys.path.insert(0, SRC_DIR)

from utils.tokenizer import mixMulanTokenizer
from model.CellTempo_VQVAE.model import VQModel
from utils.distribution import NegativeBinomial
from utils.utils_metrics import map_adata_to_reference_genes

# ──────────────────────────────────────────────────────────────
# Marker gene dict for cell-type scoring
# ──────────────────────────────────────────────────────────────
MARKER_DICT = {
    "Neuron":              ["MAP2", "TUBB3", "SYP", "DCX", "RBFOX3", "SNAP25"],
    "NPC_GPC":             ["SOX2", "NES", "VIM", "PAX6", "HES1", "FABP7"],
    "GPC_prolif":          ["MKI67", "TOP2A", "PCNA", "TUBA1B", "HMGB1", "HMGB2",
                            "H2AFZ", "TYMS", "PCLAF", "MAD2L1"],
    "Astrocyte_precursor": ["GFAP", "S100B", "ALDH1L1", "AQP4", "SLC1A3", "CD44"],
    "Astrocyte":           ["GFAP", "S100B", "ALDH1L1", "AQP4", "GJA1", "APOE"],
    "OPC":                 ["PDGFRA", "OLIG2", "SOX10", "CSPG4", "OLIG1", "NKX2-2"],
    "pre_OPC":             ["PDGFRA", "OLIG1", "GPR17", "BCAS1", "NKX2-2"],
    "Oligodendrocyte":     ["MBP", "MOG", "PLP1", "MAG", "CNP", "MOBP"],
}


def annotate_by_marker_score(raw_expr, marker_dict, reference_gene):
    """Assign cell types via marker-gene scoring (single-cell level)."""
    ad_tmp = ad.AnnData(raw_expr.copy())
    ad_tmp.var_names = reference_gene
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)

    score_cols = {}
    for ct, markers in marker_dict.items():
        existing = [g for g in markers if g in ad_tmp.var_names]
        if not existing:
            continue
        col = f"score_{ct}"
        sc.tl.score_genes(ad_tmp, gene_list=existing, score_name=col)
        score_cols[ct] = ad_tmp.obs[col].values

    score_df = pd.DataFrame(score_cols, index=np.arange(ad_tmp.n_obs))
    labels = score_df.idxmax(axis=1).values
    return labels, score_df


# ──────────────────────────────────────────────────────────────
# DEG-trajectory mode helpers
# ──────────────────────────────────────────────────────────────
def compute_deg_drug_vs_dmso(drug_name, h5ad_dir, padj_cutoff=0.05, logfc_cutoff=0.25,
                             cell_type=None):
    """Compute DEGs between drug perturbed cells and DMSO perturbed cells.
    Returns a DataFrame sorted by abs(log2FC) descending.

    If cell_type is specified, only cells matching that type (via obs['cell_type_marker'])
    are used for the comparison.
    """
    import scipy.sparse as sp

    def _load_perturbed(name):
        direct = os.path.join(h5ad_dir, f"perturb_{name}.h5ad")
        if os.path.exists(direct):
            a = sc.read_h5ad(direct)
            if sp.issparse(a.X):
                a.X = a.X.toarray()
            return a
        merged = os.path.join(h5ad_dir, f"perturb_ctrl_merged_{name}.h5ad")
        if os.path.exists(merged):
            a = sc.read_h5ad(merged)
            if sp.issparse(a.X):
                a.X = a.X.toarray()
            return a[(a.obs["condition"] == "perturbed").values].copy()
        return None

    adata_drug = _load_perturbed(drug_name)
    adata_dmso = _load_perturbed("DMSO")
    if adata_drug is None or adata_dmso is None:
        raise RuntimeError(f"Cannot load perturbed h5ad for {drug_name} or DMSO")

    if cell_type is not None:
        col = "cell_type_marker"
        if col in adata_drug.obs.columns:
            adata_drug = adata_drug[adata_drug.obs[col] == cell_type].copy()
        if col in adata_dmso.obs.columns:
            adata_dmso = adata_dmso[adata_dmso.obs[col] == cell_type].copy()
        if adata_drug.shape[0] == 0 or adata_dmso.shape[0] == 0:
            raise RuntimeError(
                f"No {cell_type} cells found for {drug_name} or DMSO after filtering"
            )

    adata_drug.obs["group"] = drug_name
    adata_dmso.obs["group"] = "DMSO"
    combined = ad.concat([adata_drug, adata_dmso], join="outer")
    combined.obs_names_make_unique()

    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.filter_genes(combined, min_cells=5)

    sc.tl.rank_genes_groups(
        combined, groupby="group", groups=[drug_name],
        reference="DMSO", method="wilcoxon",
    )

    result = combined.uns["rank_genes_groups"]
    deg_df = pd.DataFrame({
        "gene": [str(n) for n in result["names"][drug_name]],
        "log2fc": result["logfoldchanges"][drug_name].astype(float),
        "padj": result["pvals_adj"][drug_name].astype(float),
    })
    deg_df["abs_log2fc"] = deg_df["log2fc"].abs()
    deg_df["significant"] = (deg_df["padj"] < padj_cutoff) & (deg_df["abs_log2fc"] > logfc_cutoff)
    deg_df["direction"] = "ns"
    deg_df.loc[(deg_df["significant"]) & (deg_df["log2fc"] > 0), "direction"] = "up"
    deg_df.loc[(deg_df["significant"]) & (deg_df["log2fc"] < 0), "direction"] = "down"
    return deg_df.sort_values("abs_log2fc", ascending=False)


def write_perturb_gene_config(deg_df, out_path, top_n=10,
                              cluster_key="condition", cluster="perturbed",
                              amp_add_up=2, amp_mul_up=8,
                              amp_add_down=1, amp_mul_down=0):
    """Write a perturb_gene YAML config from DEG results.
    Selects top_n up-regulated and top_n down-regulated significant genes.
    Returns (up_genes, down_genes) lists.
    """
    sig = deg_df[deg_df["significant"]]
    up_genes = sig[sig["direction"] == "up"].head(top_n)["gene"].tolist()
    down_genes = sig[sig["direction"] == "down"].head(top_n)["gene"].tolist()

    module_name = "drug_deg"
    cfg = {
        "cluster_key": cluster_key,
        "gene_modules": {
            module_name: {
                "up": up_genes if up_genes else [],
                "down": down_genes if down_genes else [],
            }
        },
        "amplify_rules": [],
    }
    if up_genes:
        cfg["amplify_rules"].append({
            "cluster": cluster,
            "module": module_name,
            "direction": "up",
            "add": amp_add_up,
            "mul": amp_mul_up,
        })
    if down_genes:
        cfg["amplify_rules"].append({
            "cluster": cluster,
            "module": module_name,
            "direction": "down",
            "add": amp_add_down,
            "mul": amp_mul_down,
        })

    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return up_genes, down_genes


# ──────────────────────────────────────────────────────────────
# Helper: read target_mole.txt
# ──────────────────────────────────────────────────────────────
def load_drugs(filepath):
    drugs = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"  WARNING: skipping malformed line: {line}")
                continue
            name = parts[0].strip()
            smiles = parts[1].strip().split(" / ")[0].strip()
            drugs.append((name, smiles))
    return drugs


# ──────────────────────────────────────────────────────────────
# Helper: run generate_traj.py as subprocess
# ──────────────────────────────────────────────────────────────
def run_generation(config_path, infer_type, traj_num=0, dose=None, cuda_devices="0,1"):
    cmd = [
        sys.executable, "generate_traj.py",
        "--config_file", config_path,
        "--infer_type", infer_type,
        "--traj_num", str(traj_num),
    ]
    if dose:
        cmd.extend(["--dose", dose])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SRC_DIR, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"generate_traj.py failed (rc={result.returncode})")
    print(f"  Generation finished successfully.")


# ──────────────────────────────────────────────────────────────
# Helper: create temp YAML configs per drug
# ──────────────────────────────────────────────────────────────
def create_perturb_config(drug_name, smiles, dose, template_path, out_dir, tag=None):
    with open(template_path) as f:
        cfg = yaml.safe_load(f)
    cfg["save_name"] = f"iPSC_GPC_{drug_name}"
    cfg["smiles"] = smiles
    cfg["dose"] = dose
    if tag:
        cfg["comment"] = cfg.get("comment", "perturb_h5ad") + f"_{tag}"
    tmp = os.path.join(out_dir, f"_tmp_perturb_{drug_name}.yaml")
    with open(tmp, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp


def create_traj_config(drug_name, template_path, out_dir, celltempo_data_dir,
                       prefix_mode="paired", step_num=12,
                       h5ad_subdir="iPSC_perturb", pkl_subdir="iPSC", tag=None,
                       perturb_config_path=None, base_drug_name=None):
    """Create trajectory YAML config.

    If base_drug_name is set, use that drug's h5ad/pkl as the data source
    (used in deg_traj_mode where we run trajectory on DMSO's data with
    per-drug gene amplification via perturb_config_path).
    """
    with open(template_path) as f:
        cfg = yaml.safe_load(f)
    data_drug = base_drug_name if base_drug_name else drug_name
    h5ad_name = f"{h5ad_subdir}/perturb_ctrl_merged_{data_drug}.h5ad"
    cfg["save_name"] = f"iPSC_GPC_{drug_name}"
    cfg["dataset_names"] = [h5ad_name]
    cfg["global_dataset"] = h5ad_name
    cfg["trajectory_pkl"] = os.path.join(
        celltempo_data_dir, pkl_subdir, f"perturb_trajectory_index_{data_drug}.pkl"
    )
    cfg["wandb_log"] = False
    if perturb_config_path:
        cfg["perturb_config"] = perturb_config_path
    if tag:
        cfg["comment"] = cfg.get("comment", "traj_h5ad_perturb_drug") + f"_{tag}"
    # Steps 0,1 = prefix (ctrl, pert from h5ad), Steps 2..step_num-1 = generated
    num_generated = step_num - 2
    needed = num_generated * 29 + 60
    if cfg.get("max_new_tokens", 0) < needed:
        cfg["max_new_tokens"] = needed
    tmp = os.path.join(out_dir, f"_tmp_traj_{drug_name}.yaml")
    with open(tmp, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp


# ──────────────────────────────────────────────────────────────
# Step 2 (control): assemble trajectory without perturbation
# ──────────────────────────────────────────────────────────────
def assemble_control_trajectory(adata, celltempo_data_dir, prefix_mode="paired",
                                subsample=1.0, cell_type=None,
                                h5ad_subdir="iPSC_perturb", pkl_subdir="iPSC"):
    """Build trajectory data for the control group (no drug perturbation).

    If cell_type is specified, only cells of that type are used.
    Otherwise uses all cells (subsampled if subsample < 1.0).
    paired mode: Each cell is paired with a copy of itself (target_id=2).
    single mode: Each cell is used directly (target_id=1).
    """
    drug_name = "control"

    # Filter by cell_type if specified
    if cell_type is not None:
        type_mask = (adata.obs["cell_type_marker"] == cell_type).values
        adata_use = adata[type_mask].copy()
    else:
        adata_use = adata.copy()

    n_total = adata_use.shape[0]

    # Subsample
    if subsample < 1.0:
        np.random.seed(42)
        n_keep = max(1, int(n_total * subsample))
        idx = sorted(np.random.choice(n_total, size=n_keep, replace=False).tolist())
        adata_sub = adata_use[idx].copy()
    else:
        adata_sub = adata_use

    n_ctrl = adata_sub.shape[0]

    if prefix_mode == "single":
        # Single-cell prefix: trajectory model sees only the "perturbed" copy,
        # but h5ad stores both ctrl+copy for analysis (Step 0 & Step 1)
        adata_sub.obs["condition"] = "control"
        adata_copy = adata_sub.copy()
        adata_copy.obs["condition"] = "perturbed"

        adata_merged = ad.concat([adata_sub, adata_copy], join="outer")
        adata_merged.obs_names_make_unique()

        trajectory_list = [[n_ctrl + i] for i in range(n_ctrl)]
        target_list = [1] * n_ctrl

        save_h5ad = os.path.join(
            celltempo_data_dir, h5ad_subdir, f"perturb_ctrl_merged_{drug_name}.h5ad"
        )
        adata_merged.write_h5ad(save_h5ad)
    else:
        adata_copy = adata_sub.copy()
        adata_sub.obs["condition"] = "control"
        adata_copy.obs["condition"] = "perturbed"

        adata_merged = ad.concat([adata_sub, adata_copy], join="outer")
        adata_merged.obs_names_make_unique()

        trajectory_list = [[i, n_ctrl + i] for i in range(n_ctrl)]
        target_list = [2] * n_ctrl

        save_h5ad = os.path.join(
            celltempo_data_dir, h5ad_subdir, f"perturb_ctrl_merged_{drug_name}.h5ad"
        )
        adata_merged.write_h5ad(save_h5ad)

    traj_pkl = os.path.join(
        celltempo_data_dir, pkl_subdir, f"perturb_trajectory_index_{drug_name}.pkl"
    )
    with open(traj_pkl, "wb") as f:
        pickle.dump((trajectory_list, target_list), f, protocol=pickle.HIGHEST_PROTOCOL)

    ct_str = cell_type if cell_type else "all"
    print(f"  Control: {n_ctrl} trajectories assembled ({prefix_mode}, cell_type={ct_str}, subsample={subsample}).")
    return trajectory_list, target_list


# ──────────────────────────────────────────────────────────────
# Step 2 (drug): decode perturbation results & assemble trajectory
# ──────────────────────────────────────────────────────────────
def decode_perturb_and_assemble(drug_name, model, tokenizer, size_factors,
                                 reference_gene, adata, num_gpus,
                                 perturb_output_dir, celltempo_data_dir,
                                 prefix_mode="paired", subsample=1.0,
                                 cell_type=None,
                                 h5ad_subdir="iPSC_perturb", pkl_subdir="iPSC"):
    results = []
    task_name = f"iPSC_GPC_{drug_name}"
    for i in range(num_gpus):
        fpath = os.path.join(perturb_output_dir, f"gpu_{i}_results_{task_name}.pt")
        if os.path.exists(fpath):
            results.extend(torch.load(fpath, weights_only=False))
    if not results:
        raise RuntimeError(f"No perturbation results for {drug_name}")

    exp_gen = []
    valid_results = []
    for point in tqdm(results, desc=f"  Decoding perturb [{drug_name}]"):
        gene_name = tokenizer.decode(point["generated_ids"][-29:]).split("##")
        try:
            gene_id = [int(g) for g in gene_name[: model.num_code]]
        except (ValueError, IndexError):
            continue
        codes = model.quantize.embedding.weight[gene_id]
        quant = codes.reshape(-1)
        expr = model.decoder(quant)
        exp_gen.append(expr)
        valid_results.append(point)

    reconstruct = torch.stack(exp_gen)
    sf = torch.tensor([size_factors["brain"]] * len(exp_gen))
    fmap = F.softmax(reconstruct, dim=-1) * sf.unsqueeze(-1)
    distr = NegativeBinomial(mu=fmap, theta=torch.exp(model.theta))
    reconstruct = distr.sample_ori().detach().cpu().numpy()

    # Filter by cell_type if specified
    if cell_type is not None:
        type_mask = (adata.obs["cell_type_marker"] == cell_type).values
        type_indices = set(np.where(type_mask)[0])
        keep_gen = [i for i, p in enumerate(valid_results) if p["idx"] in type_indices]
    else:
        keep_gen = list(range(len(valid_results)))

    # Subsample if requested
    if subsample < 1.0:
        np.random.seed(42)
        n_keep = max(1, int(len(keep_gen) * subsample))
        keep_gen = sorted(np.random.choice(keep_gen, size=n_keep, replace=False).tolist())

    if not keep_gen:
        print(f"  WARNING: no cells decoded for {drug_name}, skipping.")
        return None

    reconstruct_filtered = reconstruct[keep_gen]
    results_filtered = [valid_results[i] for i in keep_gen]

    # Build ctrl adata from the corresponding original cells
    ctrl_indices = [valid_results[i]["idx"] for i in keep_gen]
    adata_ctrl = adata[ctrl_indices].copy()
    adata_gen = ad.AnnData(reconstruct_filtered)
    adata_gen.var_names = adata.var_names

    if "cell_type_marker" in adata.obs.columns:
        adata_gen.obs["cell_type_marker"] = adata.obs["cell_type_marker"].iloc[ctrl_indices].values

    adata_ctrl.obs["condition"] = "control"
    adata_gen.obs["condition"] = "perturbed"

    n_ctrl = adata_ctrl.shape[0]

    if prefix_mode == "single":
        # Single-cell prefix: only perturbed cells used by trajectory model,
        # but h5ad stores both ctrl+pert for analysis (Step 0 & Step 1)
        adata_merged = ad.concat([adata_ctrl, adata_gen], join="outer")
        adata_merged.obs_names_make_unique()

        trajectory_list = [[n_ctrl + i] for i in range(len(keep_gen))]
        target_list = [1] * len(keep_gen)

        save_h5ad = os.path.join(
            celltempo_data_dir, h5ad_subdir, f"perturb_ctrl_merged_{drug_name}.h5ad"
        )
        adata_merged.write_h5ad(save_h5ad)
    else:
        # Paired prefix: [ctrl, pert], target_id=2
        adata_merged = ad.concat([adata_ctrl, adata_gen], join="outer")
        adata_merged.obs_names_make_unique()

        trajectory_list, target_list = [], []
        for gen_i in range(len(keep_gen)):
            trajectory_list.append([gen_i, n_ctrl + gen_i])
            target_list.append(2)

        save_h5ad = os.path.join(
            celltempo_data_dir, h5ad_subdir, f"perturb_ctrl_merged_{drug_name}.h5ad"
        )
        adata_merged.write_h5ad(save_h5ad)

    traj_pkl = os.path.join(
        celltempo_data_dir, pkl_subdir, f"perturb_trajectory_index_{drug_name}.pkl"
    )
    with open(traj_pkl, "wb") as f:
        pickle.dump((trajectory_list, target_list), f, protocol=pickle.HIGHEST_PROTOCOL)

    perturb_h5ad = os.path.join(
        celltempo_data_dir, h5ad_subdir, f"perturb_{drug_name}.h5ad"
    )
    adata_gen.write_h5ad(perturb_h5ad)

    mode_str = "single-cell" if prefix_mode == "single" else "paired"
    ct_str = cell_type if cell_type else "all"
    print(f"  Assembled {len(keep_gen)} trajectories ({mode_str}, cell_type={ct_str}, subsample={subsample}) for {drug_name}.")
    return trajectory_list, target_list


# ──────────────────────────────────────────────────────────────
# Step 4: decode trajectory & compute OPC counts
# ──────────────────────────────────────────────────────────────
def decode_trajectory_and_count_opc(drug_name, model, tokenizer, size_factors,
                                     reference_gene, num_gpus, step_num,
                                     traj_output_dir, celltempo_data_dir,
                                     h5ad_subdir="iPSC_perturb", pkl_subdir="iPSC",
                                     base_drug_name=None, vqvae_decode_all=False):
    """Decode trajectory results and count cell types per step.

    When vqvae_decode_all=True, Steps 0 and 1 are also decoded through the
    scbasetraj VQVAE (from prefix VQ codes in generated_ids) instead of using
    raw h5ad expression. This ensures all steps go through the same
    reconstruction pipeline for fair comparison.
    """
    data_drug = base_drug_name if base_drug_name else drug_name
    traj_pkl_path = os.path.join(
        celltempo_data_dir, pkl_subdir, f"perturb_trajectory_index_{data_drug}.pkl"
    )
    with open(traj_pkl_path, "rb") as f:
        trajectory_list, target_id = pickle.load(f)

    results = []
    task_name = f"iPSC_GPC_{drug_name}"
    for i in range(num_gpus):
        fpath = os.path.join(traj_output_dir, f"gpu_{i}_results_{task_name}.pt")
        if os.path.exists(fpath):
            results.extend(torch.load(fpath, weights_only=False))
    if not results:
        raise RuntimeError(f"No trajectory results for {drug_name}")

    tid_val = target_id[0] if isinstance(target_id, list) else int(target_id)

    merged_h5ad_path = os.path.join(
        celltempo_data_dir, h5ad_subdir, f"perturb_ctrl_merged_{data_drug}.h5ad"
    )
    import scipy.sparse as sp
    adata_merged = sc.read_h5ad(merged_h5ad_path)
    X_merged = adata_merged.X.toarray() if sp.issparse(adata_merged.X) else adata_merged.X

    step_counts = {}

    def _vq_decode_from_generated(gen_step):
        """Decode cells for a given step from trajectory VQ codes."""
        exp_gen, sf_list = [], []
        for point in results:
            c2_start = (model.num_code + 3) * (
                gen_step + target_id[point["idx"]] - 2
            ) + 2
            if c2_start < 0 or c2_start >= len(point["generated_ids"]):
                continue
            gene_name = tokenizer.decode(point["generated_ids"][c2_start:]).split("##")
            try:
                gene_id = [int(g) for g in gene_name[: model.num_code]]
            except (ValueError, IndexError):
                continue
            codes = model.quantize.embedding.weight[gene_id]
            quant = codes.reshape(-1)
            expr = model.decoder(quant)
            exp_gen.append(expr)
            sf_list.append(size_factors["brain"])

        if not exp_gen:
            return None, "traj_pt/scbasecount"

        reconstruct = torch.stack(exp_gen)
        sf = torch.tensor(sf_list)
        fmap = F.softmax(reconstruct, dim=-1) * sf.unsqueeze(-1)
        distr = NegativeBinomial(mu=fmap, theta=torch.exp(model.theta))
        return distr.sample_ori().detach().cpu().numpy(), "traj_pt/scbasecount"

    def _vqvae_encode_decode(raw_expr):
        """Encode raw expression through VQVAE and decode back."""
        with torch.no_grad():
            x = torch.tensor(raw_expr, dtype=torch.float32)
            h = model.encode(x).latents
            dec_out = model.decode(h)
            reconstructed = dec_out.sample
            sf = torch.tensor([size_factors["brain"]] * x.shape[0])
            fmap = F.softmax(reconstructed, dim=-1) * sf.unsqueeze(-1)
            distr = NegativeBinomial(mu=fmap, theta=torch.exp(model.theta))
            return distr.sample_ori().detach().cpu().numpy()

    for gen_step in range(step_num):

        if gen_step == 0 and not vqvae_decode_all:
            ctrl_mask = (adata_merged.obs["condition"] == "control").values
            step_expr = X_merged[ctrl_mask]
            source_tag = "h5ad/control"

        elif gen_step == 1 and not vqvae_decode_all:
            pert_mask = (adata_merged.obs["condition"] == "perturbed").values
            step_expr = X_merged[pert_mask]
            source_tag = "h5ad/finetune_tahoe"

        elif gen_step <= 1 and vqvae_decode_all:
            # Try decoding prefix VQ codes from generated_ids
            step_expr, source_tag = _vq_decode_from_generated(gen_step)
            if step_expr is None:
                # Fallback: encode h5ad cells through VQVAE then decode
                # (e.g. Step 0 in single mode where prefix has no ctrl)
                if gen_step == 0:
                    ctrl_mask = (adata_merged.obs["condition"] == "control").values
                    step_expr = _vqvae_encode_decode(X_merged[ctrl_mask])
                else:
                    pert_mask = (adata_merged.obs["condition"] == "perturbed").values
                    step_expr = _vqvae_encode_decode(X_merged[pert_mask])
                source_tag = "vqvae_encode_decode"

        else:
            step_expr, source_tag = _vq_decode_from_generated(gen_step)
            if step_expr is None:
                step_counts[gen_step] = {"Total": 0}
                print(f"    Step {gen_step}: no cells decoded")
                continue

        labels, _ = annotate_by_marker_score(step_expr, MARKER_DICT, reference_gene)
        unique, counts = np.unique(labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        total = int(len(labels))
        step_result = {"Total": total}
        for ct in count_dict:
            step_result[ct] = int(count_dict[ct])
        step_counts[gen_step] = step_result

        parts = " | ".join(f"{ct}={step_result.get(ct,0)}" for ct in sorted(step_result) if ct != "Total")
        print(f"    Step {gen_step}: {total} cells ({source_tag}) | {parts}")

    return step_counts


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Drug screening pipeline for OPC differentiation")
    p.add_argument("--cuda_devices", type=str, default="0,1",
                   help="CUDA_VISIBLE_DEVICES (default: 0,1)")
    p.add_argument("--dose", type=str, default="dose_5.0",
                   choices=["dose_0.0", "dose_0.05", "dose_0.5", "dose_5.0"])
    p.add_argument("--step_num", type=int, default=12,
                   help="Number of generated trajectory steps to decode (default: 12)")
    p.add_argument("--drug", type=str, default=None,
                   help="Only run a single drug (name in target_mole.txt). "
                        "If not set, loop over all drugs.")
    p.add_argument("--no_control", action="store_true",
                   help="Skip the control group (no-perturbation baseline).")
    p.add_argument("--prefix_mode", type=str, default="paired",
                   choices=["paired", "single"],
                   help="Trajectory prefix mode: "
                        "'paired' = [ctrl, pert] (target_id=2, default); "
                        "'single' = [pert] only (target_id=1).")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. Default: CellTempo/output/celltype_counts.csv")
    p.add_argument("--skip_perturb", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip Step 1 (perturbation) if results already exist (default: True). "
                        "Use --no-skip_perturb to force re-run.")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Fraction of cells to keep in Step 2 (e.g. 0.1 = 10%%). "
                        "Default 1.0 = keep all. Applied after decoding, before trajectory assembly.")
    p.add_argument("--cell_type", type=str, default=None,
                   help="If specified, only keep cells of this type (from adata.obs['cell_type_marker']) "
                        "for trajectory generation. Default: None = use all cells.")
    p.add_argument("--no_skip_done", action="store_true",
                   help="Force re-run drugs that already have results in the output CSV.")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Tag for this experiment run. All intermediate files will be stored "
                        "in tag-suffixed directories/filenames to avoid overwriting. "
                        "E.g. --run_tag exp01 → perturb_h5ad_exp01/, traj_h5ad_perturb_drug_exp01/")
    # DEG-trajectory mode
    p.add_argument("--deg_traj_mode", action="store_true",
                   help="Enable DEG-trajectory mode: DMSO goes through normal pipeline; "
                        "other drugs use DEGs vs DMSO to define gene perturbation for trajectory. "
                        "Requires DMSO in target_mole.txt.")
    p.add_argument("--deg_top_n", type=int, default=10,
                   help="Number of top up/down DEGs to use per drug in deg_traj_mode (default: 10)")
    p.add_argument("--deg_amp_add_up", type=float, default=2,
                   help="Additive offset for up-regulated gene amplification (default: 2)")
    p.add_argument("--deg_amp_mul_up", type=float, default=8,
                   help="Multiplicative factor for up-regulated gene amplification (default: 8)")
    p.add_argument("--deg_amp_add_down", type=float, default=1,
                   help="Additive offset for down-regulated gene amplification (default: 1)")
    p.add_argument("--deg_amp_mul_down", type=float, default=0,
                   help="Multiplicative factor for down-regulated gene amplification (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    celltempo_dir = os.path.join(PROJECT_DIR, "CellTempo")
    celltempo_data_dir = os.path.join(celltempo_dir, "data")
    configs_dir = os.path.join(celltempo_dir, "configs")

    perturb_config_tmpl = os.path.join(configs_dir, "generate_perturb_tahoe_h5ad.yaml")
    traj_config_tmpl = os.path.join(configs_dir, "generate_traj_iPSC_drug.yaml")
    target_mole_file = os.path.join(celltempo_data_dir, "target_mole.txt")
    h5ad_path = os.path.join(celltempo_data_dir, "iPSC", "D30_all.h5ad")
    ref_gene_file = os.path.join(SRC_DIR, "utils", "OS_scRNA_gene_index.18791.tsv")
    sf_file = os.path.join(celltempo_data_dir, "size_factor.pkl")
    meta_info_file = os.path.join(celltempo_data_dir, "mix_meta_info_vq_traj.json")

    perturb_vq_path = (
        "/data/lep/CellTempo_noempty/ckpt/model_ckpt/"
        "vqvae_finetune_tahoe/checkpoint-200000/vqmodel"
    )
    traj_vq_path = (
        "/data/lep/CellTempo_noempty/ckpt/model_ckpt/"
        "vqvae_train_on_scbasecount/checkpoint-700000/vqmodel"
    )

    perturb_output_dir = os.path.join(celltempo_dir, "output", "perturb_h5ad")
    traj_output_dir = os.path.join(celltempo_dir, "output", "traj_h5ad_perturb_drug")

    # Apply run_tag suffix to intermediate directories
    tag = args.run_tag
    if tag:
        perturb_output_dir = perturb_output_dir + f"_{tag}"
        traj_output_dir = traj_output_dir + f"_{tag}"
        # Create tag-specific intermediate data dirs
        intermediate_h5ad_dir = os.path.join(celltempo_data_dir, f"iPSC_perturb_{tag}")
        intermediate_pkl_dir = os.path.join(celltempo_data_dir, f"iPSC_{tag}")
    else:
        intermediate_h5ad_dir = os.path.join(celltempo_data_dir, "iPSC_perturb")
        intermediate_pkl_dir = os.path.join(celltempo_data_dir, "iPSC")
    os.makedirs(perturb_output_dir, exist_ok=True)
    os.makedirs(traj_output_dir, exist_ok=True)
    os.makedirs(intermediate_h5ad_dir, exist_ok=True)
    os.makedirs(intermediate_pkl_dir, exist_ok=True)

    # Subdirectory names relative to celltempo_data_dir (used in YAML configs)
    h5ad_subdir = os.path.relpath(intermediate_h5ad_dir, celltempo_data_dir)
    pkl_subdir = os.path.relpath(intermediate_pkl_dir, celltempo_data_dir)

    # Temp config directory (tag-specific)
    tmp_config_dir = os.path.join(configs_dir, f"tmp_{tag}") if tag else configs_dir
    os.makedirs(tmp_config_dir, exist_ok=True)

    num_gpus = len(args.cuda_devices.split(","))

    # ── Load shared resources ──
    print("Loading shared resources ...")
    reference_gene = pd.read_csv(ref_gene_file, sep="\t")["gene_name"].values

    with open(sf_file, "rb") as f:
        size_factors = pickle.load(f)
    with open(meta_info_file) as f:
        meta_info = json.load(f)
    tokenizer = mixMulanTokenizer(meta_info["token_set"])

    print("Loading adata (D30_all.h5ad) ...")
    adata = sc.read_h5ad(h5ad_path)
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    sc.pp.filter_cells(adata, min_genes=200)
    adata = map_adata_to_reference_genes(adata, reference_gene)
    ref_gene_aligned = np.array(adata.var_names)

    # ── Load drug list ──
    control_only = (args.drug and args.drug.lower() == "control")
    if control_only:
        drugs = []
        print("Mode: control only (no drugs)\n")
    else:
        drugs = load_drugs(target_mole_file)
        if args.drug:
            selected = [(n, s) for n, s in drugs if n == args.drug]
            if not selected:
                print(f"Drug '{args.drug}' not found in target_mole.txt"); return
            # deg_traj_mode always needs DMSO as DEG reference
            if args.deg_traj_mode and args.drug != "DMSO":
                dmso = [(n, s) for n, s in drugs if n == "DMSO"]
                if not dmso:
                    print("ERROR: DMSO not found in target_mole.txt "
                          "(required for --deg_traj_mode)."); return
                drugs = dmso + selected
                print(f"DEG-traj single-drug mode: auto-include DMSO + {args.drug}")
            else:
                drugs = selected
        print(f"Drugs to process ({len(drugs)}): {[d[0] for d in drugs]}\n")

    # ── Prepare output CSV ──
    if args.output:
        out_csv = args.output
    elif tag:
        out_csv = os.path.join(celltempo_dir, "output", f"celltype_counts_{tag}.csv")
    else:
        out_csv = os.path.join(celltempo_dir, "output", "celltype_counts.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    steps = list(range(args.step_num))
    cell_types = list(MARKER_DICT.keys())
    csv_header = ["Drug", "Step", "Total"] + cell_types

    csv_exists = os.path.exists(out_csv)
    if not csv_exists:
        with open(out_csv, "w") as f:
            f.write(",".join(csv_header) + "\n")

    print(f"Results will be appended to: {out_csv}")
    print(f"Prefix mode: {args.prefix_mode} (target_id={'1 - perturbed cell only' if args.prefix_mode == 'single' else '2 - [ctrl, pert]'})")
    if tag:
        print(f"Run tag: {tag}")
        print(f"  Intermediate h5ad dir: {intermediate_h5ad_dir}")
        print(f"  Intermediate pkl dir:  {intermediate_pkl_dir}")
        print(f"  Perturb output dir:    {perturb_output_dir}")
        print(f"  Traj output dir:       {traj_output_dir}")
        print(f"  Temp config dir:       {tmp_config_dir}")
    print()

    # ── Load already-completed drugs from CSV to allow resuming ──
    completed_drugs = set()
    if csv_exists:
        import csv as csv_mod
        with open(out_csv, "r") as f:
            reader = csv_mod.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    completed_drugs.add(row[0])
    if completed_drugs:
        print(f"Already completed ({len(completed_drugs)}): {sorted(completed_drugs)}")
        print("  (these will be skipped; use --no-skip_done to force re-run)\n")

    # ── Helper: run Step 3 + Step 4 for a given drug_name ──
    def run_traj_and_analyze(drug_name, vqvae_decode_all=False):
        """Steps 3-4: generate trajectory & count cell types (shared by control and drugs)."""
        print(f"\n[Step 3] Trajectory generation ...")
        t_cfg = create_traj_config(
            drug_name, traj_config_tmpl, tmp_config_dir, celltempo_data_dir,
            prefix_mode=args.prefix_mode, step_num=args.step_num,
            h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir, tag=tag,
        )
        run_generation(t_cfg, "trajectory_perturb_h5ad", traj_num=0,
                       cuda_devices=args.cuda_devices)

        print(f"\n[Step 4] Trajectory analysis (cell type counting) ...")
        traj_model = VQModel.from_pretrained(
            traj_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
        )
        step_counts = decode_trajectory_and_count_opc(
            drug_name, traj_model, tokenizer, size_factors,
            ref_gene_aligned, num_gpus, args.step_num,
            traj_output_dir, celltempo_data_dir,
            h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir,
            vqvae_decode_all=vqvae_decode_all,
        )
        del traj_model
        torch.cuda.empty_cache()

        # cleanup temp config
        if os.path.exists(t_cfg):
            os.remove(t_cfg)

        return step_counts

    def append_result(drug_name, step_counts):
        """Append rows (one per step) to CSV and print."""
        all_results[drug_name] = step_counts
        with open(out_csv, "a") as f:
            for s in steps:
                sdict = step_counts.get(s, {})
                row = [drug_name, str(s), str(sdict.get("Total", 0))]
                row += [str(sdict.get(ct, 0)) for ct in cell_types]
                f.write(",".join(row) + "\n")
        print(f"\n  >> {drug_name} result appended to {out_csv}")

    # ── Loop ──
    all_results = {}
    all_names = []

    # ──────── Control group (no perturbation) ────────
    if not args.no_control:
        ctrl_name = "control"
        all_names.append(ctrl_name)
        if not args.no_skip_done and ctrl_name in completed_drugs:
            print(f"\n  [SKIP] control already in CSV, skipping.")
        else:
            print(f"\n{'=' * 70}")
            print(f"  Control group (no drug perturbation)")
            print(f"{'=' * 70}")
            try:
                print("\n[Step 1] Skipped (no perturbation)")
                print("\n[Step 2] Assembling control trajectory data ...")
                assemble_control_trajectory(adata, celltempo_data_dir,
                                            prefix_mode=args.prefix_mode,
                                            subsample=args.subsample,
                                            cell_type=args.cell_type,
                                            h5ad_subdir=h5ad_subdir,
                                            pkl_subdir=pkl_subdir)

                step_counts = run_traj_and_analyze(ctrl_name)
                append_result(ctrl_name, step_counts)
            except Exception as exc:
                import traceback
                print(f"\n  !! ERROR processing control: {exc}")
                traceback.print_exc()

    # ──────── Drug groups ────────
    if args.deg_traj_mode:
        # ══════ DEG-trajectory mode ══════
        # Phase 1: DMSO through normal pipeline
        # Phase 2: Other drugs via DEG-based gene perturbation
        print(f"\n{'#' * 70}")
        print(f"  DEG-trajectory mode  (top_n={args.deg_top_n})")
        print(f"{'#' * 70}\n")

        dmso_entry = None
        other_drugs = []
        for name, smiles in drugs:
            if name == "DMSO":
                dmso_entry = (name, smiles)
            else:
                other_drugs.append((name, smiles))

        if dmso_entry is None:
            print("ERROR: DMSO not found in drug list. Required for --deg_traj_mode.")
            return

        # ── Phase 1: DMSO full pipeline ──
        dmso_name, dmso_smiles = dmso_entry
        all_names.append(dmso_name)
        dmso_ok = True

        if not args.no_skip_done and dmso_name in completed_drugs:
            print(f"\n  [SKIP] DMSO already in CSV, skipping trajectory. "
                  f"(Perturbed h5ad still needed for DEG.)")
        else:
            print(f"\n{'=' * 70}")
            print(f"  Phase 1: DMSO (normal molecular perturbation pipeline)")
            print(f"{'=' * 70}")
            try:
                task_name = f"iPSC_GPC_{dmso_name}"
                perturb_exists = all(
                    os.path.exists(os.path.join(
                        perturb_output_dir, f"gpu_{i}_results_{task_name}.pt"))
                    for i in range(num_gpus)
                )
                if args.skip_perturb and perturb_exists:
                    print("\n[Step 1] Skipped (DMSO perturbation results exist)")
                else:
                    print("\n[Step 1] Perturbation generation (DMSO) ...")
                    p_cfg = create_perturb_config(
                        dmso_name, dmso_smiles, args.dose,
                        perturb_config_tmpl, tmp_config_dir, tag=tag,
                    )
                    run_generation(p_cfg, "perturb_h5ad", traj_num=0,
                                   dose=args.dose, cuda_devices=args.cuda_devices)
                    if os.path.exists(p_cfg):
                        os.remove(p_cfg)

                print("\n[Step 2] Decoding & assembling DMSO trajectory data ...")
                perturb_model = VQModel.from_pretrained(
                    perturb_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
                )
                ret = decode_perturb_and_assemble(
                    dmso_name, perturb_model, tokenizer, size_factors,
                    reference_gene, adata, num_gpus,
                    perturb_output_dir, celltempo_data_dir,
                    prefix_mode=args.prefix_mode,
                    subsample=args.subsample,
                    cell_type=args.cell_type,
                    h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir,
                )
                del perturb_model
                torch.cuda.empty_cache()
                if ret is None:
                    print("ERROR: Failed to decode DMSO perturbation.")
                    dmso_ok = False
                else:
                    step_counts = run_traj_and_analyze(dmso_name)
                    append_result(dmso_name, step_counts)
            except Exception as exc:
                import traceback
                print(f"\n  !! ERROR processing DMSO: {exc}")
                traceback.print_exc()
                dmso_ok = False

        # Ensure DMSO perturbed h5ad exists (needed for DEG computation)
        dmso_perturb_h5ad = os.path.join(intermediate_h5ad_dir, f"perturb_DMSO.h5ad")
        if not os.path.exists(dmso_perturb_h5ad):
            dmso_merged = os.path.join(intermediate_h5ad_dir, f"perturb_ctrl_merged_DMSO.h5ad")
            if not os.path.exists(dmso_merged):
                print("ERROR: Neither perturb_DMSO.h5ad nor merged DMSO h5ad found. "
                      "Cannot compute DEGs.")
                dmso_ok = False

        if not dmso_ok:
            print("Aborting DEG-trajectory mode due to DMSO processing failure.")
            return

        # ── Prepare control base data for trajectory generation ──
        ctrl_merged = os.path.join(
            intermediate_h5ad_dir, "perturb_ctrl_merged_control.h5ad")
        ctrl_pkl = os.path.join(
            intermediate_pkl_dir, "perturb_trajectory_index_control.pkl")
        if not os.path.exists(ctrl_merged) or not os.path.exists(ctrl_pkl):
            print("\n[Base] Assembling control (original) trajectory data ...")
            assemble_control_trajectory(
                adata, celltempo_data_dir,
                prefix_mode=args.prefix_mode,
                subsample=args.subsample,
                cell_type=args.cell_type,
                h5ad_subdir=h5ad_subdir,
                pkl_subdir=pkl_subdir,
            )
        else:
            print("\n[Base] Control trajectory data already exists, reusing.")

        # ── Phase 2: Other drugs (DEG-based gene perturbation) ──
        deg_config_dir = os.path.join(tmp_config_dir, "deg_perturb_configs")
        os.makedirs(deg_config_dir, exist_ok=True)

        for drug_name, smiles in other_drugs:
            all_names.append(drug_name)
            if not args.no_skip_done and drug_name in completed_drugs:
                print(f"\n  [SKIP] {drug_name} already in CSV, skipping.")
                continue

            print(f"\n{'=' * 70}")
            print(f"  Phase 2: {drug_name}  (DEG-based trajectory)")
            print(f"{'=' * 70}")

            try:
                # Step 1: molecular perturbation (to get perturbed cells for DEG)
                task_name = f"iPSC_GPC_{drug_name}"
                perturb_exists = all(
                    os.path.exists(os.path.join(
                        perturb_output_dir, f"gpu_{i}_results_{task_name}.pt"))
                    for i in range(num_gpus)
                )
                if args.skip_perturb and perturb_exists:
                    print("\n[Step 1] Skipped (perturbation results exist)")
                else:
                    print("\n[Step 1] Perturbation generation ...")
                    p_cfg = create_perturb_config(
                        drug_name, smiles, args.dose,
                        perturb_config_tmpl, tmp_config_dir, tag=tag,
                    )
                    run_generation(p_cfg, "perturb_h5ad", traj_num=0,
                                   dose=args.dose, cuda_devices=args.cuda_devices)
                    if os.path.exists(p_cfg):
                        os.remove(p_cfg)

                # Step 2: decode perturbation to get perturb_{drug}.h5ad
                drug_perturb_h5ad = os.path.join(
                    intermediate_h5ad_dir, f"perturb_{drug_name}.h5ad")
                if not os.path.exists(drug_perturb_h5ad):
                    print("\n[Step 2] Decoding perturbation (for DEG analysis) ...")
                    perturb_model = VQModel.from_pretrained(
                        perturb_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
                    )
                    decode_perturb_and_assemble(
                        drug_name, perturb_model, tokenizer, size_factors,
                        reference_gene, adata, num_gpus,
                        perturb_output_dir, celltempo_data_dir,
                        prefix_mode=args.prefix_mode,
                        subsample=args.subsample,
                        cell_type=args.cell_type,
                        h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir,
                    )
                    del perturb_model
                    torch.cuda.empty_cache()
                else:
                    print(f"\n[Step 2] Skipped (perturb_{drug_name}.h5ad exists)")

                # Step 2b: DEG analysis (drug vs DMSO)
                print(f"\n[DEG] Computing DEGs: {drug_name} vs DMSO ...")
                deg_df = compute_deg_drug_vs_dmso(drug_name, intermediate_h5ad_dir)

                n_up = int((deg_df["direction"] == "up").sum())
                n_down = int((deg_df["direction"] == "down").sum())
                print(f"  Significant DEGs: {n_up} up, {n_down} down")

                deg_csv = os.path.join(deg_config_dir, f"deg_{drug_name}_vs_DMSO.csv")
                deg_df.to_csv(deg_csv, index=False)

                # Write per-drug perturb_gene config
                perturb_gene_cfg = os.path.join(
                    deg_config_dir, f"perturb_gene_{drug_name}.yaml")
                up_genes, down_genes = write_perturb_gene_config(
                    deg_df, perturb_gene_cfg,
                    top_n=args.deg_top_n,
                    amp_add_up=args.deg_amp_add_up,
                    amp_mul_up=args.deg_amp_mul_up,
                    amp_add_down=args.deg_amp_add_down,
                    amp_mul_down=args.deg_amp_mul_down,
                )
                print(f"  Top-{args.deg_top_n} UP:   {up_genes}")
                print(f"  Top-{args.deg_top_n} DOWN: {down_genes}")
                print(f"  Config: {perturb_gene_cfg}")

                if not up_genes and not down_genes:
                    print(f"  WARNING: no significant DEGs, skipping trajectory.")
                    continue

                # Step 3: trajectory (original cells + per-drug gene amplification)
                print(f"\n[Step 3] Trajectory generation "
                      f"(control base + {drug_name} DEG amplification) ...")
                t_cfg = create_traj_config(
                    drug_name, traj_config_tmpl, tmp_config_dir,
                    celltempo_data_dir,
                    prefix_mode=args.prefix_mode, step_num=args.step_num,
                    h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir, tag=tag,
                    perturb_config_path=perturb_gene_cfg,
                    base_drug_name="control",
                )
                run_generation(t_cfg, "trajectory_perturb_h5ad", traj_num=0,
                               cuda_devices=args.cuda_devices)

                # Step 4: decode trajectory & count cell types
                print(f"\n[Step 4] Trajectory analysis (cell type counting) ...")
                traj_model = VQModel.from_pretrained(
                    traj_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
                )
                step_counts = decode_trajectory_and_count_opc(
                    drug_name, traj_model, tokenizer, size_factors,
                    ref_gene_aligned, num_gpus, args.step_num,
                    traj_output_dir, celltempo_data_dir,
                    h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir,
                    base_drug_name="control",
                    vqvae_decode_all=True,
                )
                del traj_model
                torch.cuda.empty_cache()

                if os.path.exists(t_cfg):
                    os.remove(t_cfg)

                append_result(drug_name, step_counts)

            except Exception as exc:
                import traceback
                print(f"\n  !! ERROR processing {drug_name}: {exc}")
                traceback.print_exc()
                continue

    else:
        # ══════ Default mode: molecular perturbation pipeline ══════
        for drug_name, smiles in drugs:
            all_names.append(drug_name)
            if not args.no_skip_done and drug_name in completed_drugs:
                print(f"\n  [SKIP] {drug_name} already in CSV, skipping.")
                continue
            print(f"\n{'=' * 70}")
            print(f"  Drug: {drug_name}   SMILES: {smiles}")
            print(f"{'=' * 70}")

            try:
                # ---- Step 1: perturbation ----
                task_name = f"iPSC_GPC_{drug_name}"
                perturb_exists = all(
                    os.path.exists(os.path.join(
                        perturb_output_dir, f"gpu_{i}_results_{task_name}.pt"))
                    for i in range(num_gpus)
                )
                if args.skip_perturb and perturb_exists:
                    print("\n[Step 1] Skipped (perturbation results already exist)")
                else:
                    print("\n[Step 1] Perturbation generation ...")
                    p_cfg = create_perturb_config(
                        drug_name, smiles, args.dose,
                        perturb_config_tmpl, tmp_config_dir, tag=tag,
                    )
                    run_generation(p_cfg, "perturb_h5ad", traj_num=0,
                                   dose=args.dose, cuda_devices=args.cuda_devices)
                    if os.path.exists(p_cfg):
                        os.remove(p_cfg)

                # ---- Step 2: assemble trajectory data ----
                print("\n[Step 2] Decoding & assembling trajectory data ...")
                perturb_model = VQModel.from_pretrained(
                    perturb_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
                )
                ret = decode_perturb_and_assemble(
                    drug_name, perturb_model, tokenizer, size_factors,
                    reference_gene, adata, num_gpus,
                    perturb_output_dir, celltempo_data_dir,
                    prefix_mode=args.prefix_mode,
                    subsample=args.subsample,
                    cell_type=args.cell_type,
                    h5ad_subdir=h5ad_subdir, pkl_subdir=pkl_subdir,
                )
                del perturb_model
                torch.cuda.empty_cache()
                if ret is None:
                    continue

                # ---- Steps 3-4 ----
                step_counts = run_traj_and_analyze(drug_name)
                append_result(drug_name, step_counts)

            except Exception as exc:
                import traceback
                print(f"\n  !! ERROR processing {drug_name}: {exc}")
                traceback.print_exc()
                continue

    # ──────────────────────────────────────────────────────────
    # Final summary (print all collected results)
    # ──────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results collected."); return

    rows = []
    for name in all_names:
        if name not in all_results:
            continue
        for s in steps:
            sdict = all_results[name].get(s, {})
            row = {"Drug": name, "Step": s, "Total": sdict.get("Total", 0)}
            for ct in cell_types:
                row[ct] = sdict.get(ct, 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 70}")
    print("Cell Type Counts Summary")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print(f"\nAll results saved in {out_csv}")


if __name__ == "__main__":
    main()
