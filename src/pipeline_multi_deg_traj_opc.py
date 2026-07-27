#!/usr/bin/env python
"""
Multi-molecule union DEG trajectory pipeline for OPC differentiation analysis.

Workflow:
  1. DMSO perturbation generation → decode → perturbed h5ad
  2. Each input drug: perturbation generation → decode → perturbed h5ad
  3. DEG analysis: each drug vs DMSO (Wilcoxon)
  4. Take union of top-N significant up/down DEGs across all drugs
  5. Assemble original (control) cells as trajectory base
  6. Generate trajectory from original cells with union-DEG gene amplification
  7. Decode trajectory VQ tokens, marker-gene scoring, count cell types per step

Key difference from single-drug deg_traj_mode:
  Instead of running a separate trajectory per drug, this script combines
  DEGs from multiple drugs into one union set, perturbs original cells once,
  and generates a single trajectory.

Input drug list:
  - --drug_file: file with drug names (one per line, or tab-separated with SMILES)
  - --drugs: comma-separated drug names (SMILES looked up from target_mole.txt)
  DMSO must be available in target_mole.txt as the reference.
"""

import os
import sys
import json
import yaml
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = SCRIPT_DIR
CELLTEMPO_DIR = os.path.dirname(SRC_DIR)
PROJECT_DIR = os.path.dirname(CELLTEMPO_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pipeline_drug_screen_opc import (
    MARKER_DICT,
    annotate_by_marker_score,
    compute_deg_drug_vs_dmso,
    load_drugs,
    run_generation,
    create_perturb_config,
    create_traj_config,
    assemble_control_trajectory,
    decode_perturb_and_assemble,
    decode_trajectory_and_count_opc,
)

from utils.tokenizer import mixMulanTokenizer
from model.CellTempo_VQVAE.model import VQModel
from utils.utils_metrics import map_adata_to_reference_genes


# ──────────────────────────────────────────────────────────────
# Load input drug list (supports name-only or name+SMILES)
# ──────────────────────────────────────────────────────────────
def load_input_drugs(filepath, drug_map=None):
    """Load drug names from a file.

    Supports:
      - One drug name per line (SMILES looked up from drug_map)
      - Tab-separated: drug_name<TAB>SMILES
    Lines starting with '#' are ignored.
    """
    drugs = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                name = parts[0].strip()
                smiles = parts[1].strip().split(" / ")[0].strip()
                drugs.append((name, smiles))
            else:
                name = parts[0].strip()
                if drug_map and name in drug_map:
                    drugs.append((name, drug_map[name]))
                else:
                    print(f"  WARNING: cannot resolve SMILES for '{name}', skipping")
    return drugs


# ──────────────────────────────────────────────────────────────
# Write union perturb_gene config from multiple drugs' DEGs
# ──────────────────────────────────────────────────────────────
def write_union_perturb_gene_config(drug_deg_pairs, out_path, top_n=10,
                                    cluster_key="condition", cluster="perturbed",
                                    amp_add_up=2, amp_mul_up=8,
                                    amp_add_down=1, amp_mul_down=0):
    """Write a perturb_gene YAML from the union of top-N DEGs across drugs.

    For each drug, selects top-N significant up/down DEGs (already sorted by
    abs_log2fc from compute_deg_drug_vs_dmso), then takes the union.
    Genes appearing in both up and down are removed to avoid conflict.

    Returns (up_genes, down_genes, per_drug_info_dict).
    """
    all_up = set()
    all_down = set()
    per_drug = {}

    for drug_name, deg_df in drug_deg_pairs:
        sig = deg_df[deg_df["significant"]]
        up = sig[sig["direction"] == "up"].head(top_n)["gene"].tolist()
        down = sig[sig["direction"] == "down"].head(top_n)["gene"].tolist()
        all_up.update(up)
        all_down.update(down)
        per_drug[drug_name] = {"up": up, "down": down}

    conflict = all_up & all_down
    if conflict:
        print(f"  WARNING: {len(conflict)} genes in both up & down sets, "
              f"removing: {sorted(conflict)}")
        all_up -= conflict
        all_down -= conflict

    up_genes = sorted(all_up)
    down_genes = sorted(all_down)

    module_name = "multi_drug_union_deg"
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

    return up_genes, down_genes, per_drug


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-molecule union DEG trajectory pipeline for OPC differentiation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--drug_file", type=str,
                   help="File with drug names (one per line, or tab-separated with SMILES). "
                        "DMSO is automatically included as the reference.")
    g.add_argument("--drugs", type=str,
                   help="Comma-separated drug names (SMILES from target_mole.txt). "
                        "DMSO is automatically included as the reference.")

    p.add_argument("--combo_name", type=str, default=None,
                   help="Name for the combined multi-drug result. "
                        "Default: auto-generated from drug names.")

    p.add_argument("--cuda_devices", type=str, default="0,1",
                   help="CUDA_VISIBLE_DEVICES (default: 0,1)")
    p.add_argument("--dose", type=str, default="dose_5.0",
                   choices=["dose_0.0", "dose_0.05", "dose_0.5", "dose_5.0"])
    p.add_argument("--step_num", type=int, default=12,
                   help="Number of trajectory steps to decode (default: 12)")
    p.add_argument("--prefix_mode", type=str, default="paired",
                   choices=["paired", "single"],
                   help="Trajectory prefix mode: 'paired' or 'single' (default: paired)")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. Default: CellTempo/output/celltype_counts_multi_deg_{combo_name}.csv")
    p.add_argument("--skip_perturb", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip perturbation if results exist (default: True)")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Fraction of cells to keep (default: 1.0 = all)")
    p.add_argument("--cell_type", type=str, default=None,
                   help="Filter cells by this type (from adata.obs['cell_type_marker'])")
    p.add_argument("--h5ad_path", type=str, default=None,
                   help="Path to the starting h5ad file for trajectory generation (default: "
                        "CellTempo/data/iPSC/D30_all.h5ad). The h5ad must have var_names matching "
                        "the 18791-gene reference, or be mappable to it via map_adata_to_reference_genes.")
    p.add_argument("--perturb_adata_path", type=str, default=None,
                   help="Path to the adata used for perturbation decode (default: same as --h5ad_path "
                        "or D30_all.h5ad). Set this to the ORIGINAL D30_all.h5ad when --h5ad_path points "
                        "to a different file (e.g. step11 output), so that perturbation .pt indices match.")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Tag for combo-specific results (trajectory, DEG, CSV). "
                        "Perturbation results are always shared across runs.")
    p.add_argument("--no_control", action="store_true",
                   help="Skip running a separate control (no-perturbation) trajectory")

    p.add_argument("--deg_top_n", type=int, default=10,
                   help="Top-N up/down DEGs per drug before union (default: 10)")
    p.add_argument("--deg_amp_add_up", type=float, default=2,
                   help="Additive offset for up-regulated gene amplification (default: 2)")
    p.add_argument("--deg_amp_mul_up", type=float, default=8,
                   help="Multiplicative factor for up-regulated gene amplification (default: 8)")
    p.add_argument("--deg_amp_add_down", type=float, default=1,
                   help="Additive offset for down-regulated gene amplification (default: 1)")
    p.add_argument("--deg_amp_mul_down", type=float, default=0,
                   help="Multiplicative factor for down-regulated gene amplification (default: 0)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Paths ──
    celltempo_dir = os.path.join(PROJECT_DIR, "CellTempo")
    celltempo_data_dir = os.path.join(celltempo_dir, "data")
    configs_dir = os.path.join(celltempo_dir, "configs")

    perturb_config_tmpl = os.path.join(configs_dir, "generate_perturb_tahoe_h5ad.yaml")
    traj_config_tmpl = os.path.join(configs_dir, "generate_traj_iPSC_drug.yaml")
    target_mole_file = os.path.join(celltempo_data_dir, "target_mole.txt")
    if args.h5ad_path:
        h5ad_path = os.path.abspath(os.path.expanduser(args.h5ad_path))
        if not os.path.isfile(h5ad_path):
            raise FileNotFoundError(f"Input h5ad not found: {h5ad_path}")
        print(f"Using custom input h5ad: {h5ad_path}")
    else:
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

    # ── Shared .pt directory (GPU generation results, cell_type-independent) ──
    perturb_output_dir = os.path.join(celltempo_dir, "output", "perturb_h5ad")

    # ── Per-cell-type directories for decoded h5ad/pkl (reusable across combos) ──
    ct_label = args.cell_type if args.cell_type else "all"
    shared_h5ad_dir = os.path.join(celltempo_data_dir, f"iPSC_perturb_{ct_label}")
    shared_pkl_dir = os.path.join(celltempo_data_dir, f"iPSC_{ct_label}")
    perturb_h5ad_subdir = f"iPSC_perturb_{ct_label}"
    perturb_pkl_subdir = f"iPSC_{ct_label}"

    # ── Combo-specific directories (trajectory, control assembly, DEG configs) ──
    tag = args.run_tag
    traj_output_dir = os.path.join(celltempo_dir, "output", "traj_h5ad_perturb_drug")
    if tag:
        traj_output_dir += f"_{tag}"
        combo_h5ad_dir = os.path.join(celltempo_data_dir, f"iPSC_perturb_{tag}")
        combo_pkl_dir = os.path.join(celltempo_data_dir, f"iPSC_{tag}")
    else:
        combo_h5ad_dir = shared_h5ad_dir
        combo_pkl_dir = shared_pkl_dir
    combo_h5ad_subdir = os.path.relpath(combo_h5ad_dir, celltempo_data_dir)
    combo_pkl_subdir = os.path.relpath(combo_pkl_dir, celltempo_data_dir)

    for d in [perturb_output_dir, traj_output_dir,
              shared_h5ad_dir, shared_pkl_dir,
              combo_h5ad_dir, combo_pkl_dir]:
        os.makedirs(d, exist_ok=True)

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

    print(f"Loading adata ({os.path.basename(h5ad_path)}) ...")
    adata = sc.read_h5ad(h5ad_path)
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    sc.pp.filter_cells(adata, min_genes=200)
    adata = map_adata_to_reference_genes(adata, reference_gene)
    ref_gene_aligned = np.array(adata.var_names)

    # ── Load separate adata for perturbation decode if specified ──
    # When --h5ad_path points to a non-original file (e.g. step11 output),
    # perturbation .pt indices won't match. Use --perturb_adata_path to
    # load the original D30_all.h5ad for perturbation decode.
    if args.perturb_adata_path:
        perturb_adata_path = os.path.abspath(os.path.expanduser(args.perturb_adata_path))
        if not os.path.isfile(perturb_adata_path):
            raise FileNotFoundError(f"Perturb adata not found: {perturb_adata_path}")
        print(f"Loading perturb adata ({os.path.basename(perturb_adata_path)}) ...")
        perturb_adata = sc.read_h5ad(perturb_adata_path)
        perturb_adata = perturb_adata[:, ~perturb_adata.var_names.duplicated()].copy()
        sc.pp.filter_cells(perturb_adata, min_genes=200)
        perturb_adata = map_adata_to_reference_genes(perturb_adata, reference_gene)
        print(f"  perturb_adata: {perturb_adata.shape} (used for decode_perturb_and_assemble)")
    else:
        perturb_adata = adata

    # ── Resolve input drug list ──
    all_mole = load_drugs(target_mole_file)
    drug_map = {n: s for n, s in all_mole}

    if args.drug_file:
        input_drugs = load_input_drugs(args.drug_file, drug_map=drug_map)
    else:
        names = [d.strip() for d in args.drugs.split(",")]
        input_drugs = []
        for n in names:
            if n in drug_map:
                input_drugs.append((n, drug_map[n]))
            else:
                print(f"  WARNING: '{n}' not in target_mole.txt, skipping")

    non_dmso_drugs = [(n, s) for n, s in input_drugs if n != "DMSO"]
    if not non_dmso_drugs:
        print("ERROR: no non-DMSO drugs specified."); return

    if "DMSO" not in drug_map:
        print("ERROR: DMSO not found in target_mole.txt. Required as reference."); return
    dmso_smiles = drug_map["DMSO"]

    combo_name = args.combo_name
    if not combo_name:
        if len(non_dmso_drugs) <= 3:
            combo_name = "_".join(n for n, _ in non_dmso_drugs)
        else:
            combo_name = f"multi{len(non_dmso_drugs)}drugs"

    print(f"\n{'#' * 70}")
    print(f"  Multi-molecule union DEG trajectory pipeline")
    print(f"  Drugs: {[n for n, _ in non_dmso_drugs]}")
    print(f"  Combo name: {combo_name}")
    print(f"  DEG top-N per drug: {args.deg_top_n}")
    print(f"{'#' * 70}\n")

    # ── Output CSV ──
    if args.output:
        out_csv = args.output
    elif tag:
        out_csv = os.path.join(
            celltempo_dir, "output",
            f"celltype_counts_multi_deg_{combo_name}_{tag}.csv")
    else:
        out_csv = os.path.join(
            celltempo_dir, "output",
            f"celltype_counts_multi_deg_{combo_name}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    steps = list(range(args.step_num))
    cell_types = list(MARKER_DICT.keys())
    csv_header = ["Drug", "Step", "Total"] + cell_types

    if not os.path.exists(out_csv):
        with open(out_csv, "w") as f:
            f.write(",".join(csv_header) + "\n")

    print(f"Results will be saved to: {out_csv}")
    print(f"Prefix mode: {args.prefix_mode}")
    print(f"Perturbation results (shared): {perturb_output_dir}")
    print(f"Perturbation h5ad   (shared): {shared_h5ad_dir}")
    if tag:
        print(f"Run tag: {tag}")
        print(f"Combo results (tag-specific): {traj_output_dir}")
    print()

    def append_result(name, step_counts):
        with open(out_csv, "a") as f:
            for s in steps:
                sdict = step_counts.get(s, {})
                row = [name, str(s), str(sdict.get("Total", 0))]
                row += [str(sdict.get(ct, 0)) for ct in cell_types]
                f.write(",".join(row) + "\n")
        print(f"\n  >> {name} result appended to {out_csv}")

    # ══════════════════════════════════════════════════════════
    # Phase 1: DMSO perturbation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 1: DMSO perturbation generation")
    print(f"{'=' * 70}")

    task_name_dmso = "iPSC_GPC_DMSO"
    dmso_perturb_exists = all(
        os.path.exists(os.path.join(
            perturb_output_dir, f"gpu_{i}_results_{task_name_dmso}.pt"))
        for i in range(num_gpus)
    )
    if args.skip_perturb and dmso_perturb_exists:
        print("\n[Step 1] Skipped (DMSO perturbation results exist)")
    else:
        print("\n[Step 1] DMSO perturbation generation ...")
        p_cfg = create_perturb_config(
            "DMSO", dmso_smiles, args.dose,
            perturb_config_tmpl, tmp_config_dir, tag=None,
        )
        run_generation(p_cfg, "perturb_h5ad", traj_num=0,
                       dose=args.dose, cuda_devices=args.cuda_devices)
        if os.path.exists(p_cfg):
            os.remove(p_cfg)

    dmso_perturb_h5ad = os.path.join(shared_h5ad_dir, "perturb_DMSO.h5ad")
    if not os.path.exists(dmso_perturb_h5ad):
        print("\n[Step 2] Decoding DMSO perturbation ...")
        perturb_model = VQModel.from_pretrained(
            perturb_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
        )
        decode_perturb_and_assemble(
            "DMSO", perturb_model, tokenizer, size_factors,
            reference_gene, perturb_adata, num_gpus,
            perturb_output_dir, celltempo_data_dir,
            prefix_mode=args.prefix_mode,
            subsample=args.subsample,
            cell_type=args.cell_type,
            h5ad_subdir=perturb_h5ad_subdir, pkl_subdir=perturb_pkl_subdir,
        )
        del perturb_model
        torch.cuda.empty_cache()
    else:
        print(f"\n[Step 2] Skipped (perturb_DMSO.h5ad exists)")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Each drug perturbation (for DEG computation)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 2: Drug perturbation generation (for DEG analysis)")
    print(f"{'=' * 70}")

    for drug_name, smiles in non_dmso_drugs:
        print(f"\n--- {drug_name} ---")

        task_name = f"iPSC_GPC_{drug_name}"
        perturb_exists = all(
            os.path.exists(os.path.join(
                perturb_output_dir, f"gpu_{i}_results_{task_name}.pt"))
            for i in range(num_gpus)
        )
        if args.skip_perturb and perturb_exists:
            print(f"  [Step 1] Skipped (perturbation results exist)")
        else:
            print(f"  [Step 1] Perturbation generation ...")
            p_cfg = create_perturb_config(
                drug_name, smiles, args.dose,
                perturb_config_tmpl, tmp_config_dir, tag=None,
            )
            run_generation(p_cfg, "perturb_h5ad", traj_num=0,
                           dose=args.dose, cuda_devices=args.cuda_devices)
            if os.path.exists(p_cfg):
                os.remove(p_cfg)

        drug_perturb_h5ad = os.path.join(
            shared_h5ad_dir, f"perturb_{drug_name}.h5ad")
        if not os.path.exists(drug_perturb_h5ad):
            print(f"  [Step 2] Decoding perturbation ...")
            perturb_model = VQModel.from_pretrained(
                perturb_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
            )
            decode_perturb_and_assemble(
                drug_name, perturb_model, tokenizer, size_factors,
                reference_gene, perturb_adata, num_gpus,
                perturb_output_dir, celltempo_data_dir,
                prefix_mode=args.prefix_mode,
                subsample=args.subsample,
                cell_type=args.cell_type,
                h5ad_subdir=perturb_h5ad_subdir, pkl_subdir=perturb_pkl_subdir,
            )
            del perturb_model
            torch.cuda.empty_cache()
        else:
            print(f"  [Step 2] Skipped (perturb_{drug_name}.h5ad exists)")

    # ══════════════════════════════════════════════════════════
    # Phase 3: DEG analysis (each drug vs DMSO) + union
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 3: DEG analysis & union")
    print(f"{'=' * 70}")

    deg_config_dir = os.path.join(tmp_config_dir, "deg_perturb_configs")
    os.makedirs(deg_config_dir, exist_ok=True)

    drug_deg_pairs = []
    for drug_name, _ in non_dmso_drugs:
        print(f"\n  [DEG] {drug_name} vs DMSO ...")
        deg_df = compute_deg_drug_vs_dmso(drug_name, shared_h5ad_dir)

        n_up = int((deg_df["direction"] == "up").sum())
        n_down = int((deg_df["direction"] == "down").sum())
        print(f"    Significant DEGs: {n_up} up, {n_down} down")

        deg_csv = os.path.join(deg_config_dir, f"deg_{drug_name}_vs_DMSO.csv")
        deg_df.to_csv(deg_csv, index=False)
        drug_deg_pairs.append((drug_name, deg_df))

    # Union of DEGs
    print(f"\n  [UNION] Computing union of top-{args.deg_top_n} DEGs across "
          f"{len(non_dmso_drugs)} drugs ...")
    union_cfg_path = os.path.join(
        deg_config_dir, f"perturb_gene_union_{combo_name}.yaml")
    up_genes, down_genes, per_drug_info = write_union_perturb_gene_config(
        drug_deg_pairs, union_cfg_path,
        top_n=args.deg_top_n,
        amp_add_up=args.deg_amp_add_up,
        amp_mul_up=args.deg_amp_mul_up,
        amp_add_down=args.deg_amp_add_down,
        amp_mul_down=args.deg_amp_mul_down,
    )

    print(f"\n  Per-drug DEG contributions:")
    for drug_name, info in per_drug_info.items():
        print(f"    {drug_name}: UP={info['up']}  DOWN={info['down']}")

    print(f"\n  Union UP  ({len(up_genes)} genes): {up_genes}")
    print(f"  Union DOWN ({len(down_genes)} genes): {down_genes}")
    print(f"  Config saved: {union_cfg_path}")

    if not up_genes and not down_genes:
        print("\n  ERROR: no significant DEGs in union. Cannot generate trajectory.")
        return

    # Save union gene summary
    union_summary = {
        "combo_name": combo_name,
        "drugs": [n for n, _ in non_dmso_drugs],
        "deg_top_n": args.deg_top_n,
        "union_up": up_genes,
        "union_down": down_genes,
        "per_drug": per_drug_info,
    }
    summary_path = os.path.join(deg_config_dir, f"union_deg_summary_{combo_name}.json")
    with open(summary_path, "w") as f:
        json.dump(union_summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved: {summary_path}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Assemble control (original) trajectory base
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 4: Assemble control (original cells) trajectory base")
    print(f"{'=' * 70}")

    ctrl_merged = os.path.join(
        combo_h5ad_dir, "perturb_ctrl_merged_control.h5ad")
    ctrl_pkl = os.path.join(
        combo_pkl_dir, "perturb_trajectory_index_control.pkl")

    if not os.path.exists(ctrl_merged) or not os.path.exists(ctrl_pkl):
        print("\n  Assembling control trajectory data ...")
        assemble_control_trajectory(
            adata, celltempo_data_dir,
            prefix_mode=args.prefix_mode,
            subsample=args.subsample,
            cell_type=args.cell_type,
            h5ad_subdir=combo_h5ad_subdir,
            pkl_subdir=combo_pkl_subdir,
        )
    else:
        print("  Control trajectory data already exists, reusing.")

    # ══════════════════════════════════════════════════════════
    # (Optional) Control trajectory without gene amplification
    # ══════════════════════════════════════════════════════════
    if not args.no_control:
        print(f"\n{'=' * 70}")
        print(f"  Control trajectory (no gene amplification, for comparison)")
        print(f"{'=' * 70}")
        try:
            print("\n[Step 3] Trajectory generation (control) ...")
            t_cfg = create_traj_config(
                "control", traj_config_tmpl, tmp_config_dir, celltempo_data_dir,
                prefix_mode=args.prefix_mode, step_num=args.step_num,
                h5ad_subdir=combo_h5ad_subdir, pkl_subdir=combo_pkl_subdir, tag=tag,
            )
            run_generation(t_cfg, "trajectory_perturb_h5ad", traj_num=0,
                           cuda_devices=args.cuda_devices)

            print("\n[Step 4] Trajectory analysis (control) ...")
            traj_model = VQModel.from_pretrained(
                traj_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
            )
            step_counts = decode_trajectory_and_count_opc(
                "control", traj_model, tokenizer, size_factors,
                ref_gene_aligned, num_gpus, args.step_num,
                traj_output_dir, celltempo_data_dir,
                h5ad_subdir=combo_h5ad_subdir, pkl_subdir=combo_pkl_subdir,
                vqvae_decode_all=False,
            )
            del traj_model
            torch.cuda.empty_cache()

            if os.path.exists(t_cfg):
                os.remove(t_cfg)

            append_result("control", step_counts)
        except Exception as exc:
            import traceback
            print(f"\n  !! ERROR processing control: {exc}")
            traceback.print_exc()

    # ══════════════════════════════════════════════════════════
    # Phase 5: Trajectory with union-DEG gene amplification
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 5: Trajectory generation ({combo_name}, union DEG amplification)")
    print(f"{'=' * 70}")

    print(f"\n[Step 3] Trajectory generation (control base + union DEG) ...")
    t_cfg = create_traj_config(
        combo_name, traj_config_tmpl, tmp_config_dir, celltempo_data_dir,
        prefix_mode=args.prefix_mode, step_num=args.step_num,
        h5ad_subdir=combo_h5ad_subdir, pkl_subdir=combo_pkl_subdir, tag=tag,
        perturb_config_path=union_cfg_path,
        base_drug_name="control",
    )
    run_generation(t_cfg, "trajectory_perturb_h5ad", traj_num=0,
                   cuda_devices=args.cuda_devices)

    # ══════════════════════════════════════════════════════════
    # Phase 6: Decode trajectory & count cell types
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Phase 6: Trajectory analysis ({combo_name})")
    print(f"{'=' * 70}")

    print(f"\n[Step 4] Decoding trajectory & counting cell types ...")
    traj_model = VQModel.from_pretrained(
        traj_vq_path, cvq_distance="cos", cvq_anchor="probrandom"
    )
    step_counts = decode_trajectory_and_count_opc(
        combo_name, traj_model, tokenizer, size_factors,
        ref_gene_aligned, num_gpus, args.step_num,
        traj_output_dir, celltempo_data_dir,
        h5ad_subdir=combo_h5ad_subdir, pkl_subdir=combo_pkl_subdir,
        base_drug_name="control",
        vqvae_decode_all=True,
    )
    del traj_model
    torch.cuda.empty_cache()

    if os.path.exists(t_cfg):
        os.remove(t_cfg)

    append_result(combo_name, step_counts)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Summary: {combo_name}")
    print(f"{'=' * 70}")
    print(f"  Drugs combined: {[n for n, _ in non_dmso_drugs]}")
    print(f"  Union UP genes  ({len(up_genes)}): {up_genes}")
    print(f"  Union DOWN genes ({len(down_genes)}): {down_genes}")
    print()

    rows = []
    for s in steps:
        sdict = step_counts.get(s, {})
        row = {"Step": s, "Total": sdict.get("Total", 0)}
        for ct in cell_types:
            row[ct] = sdict.get(ct, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nResults saved to: {out_csv}")
    print(f"Union DEG config: {union_cfg_path}")
    print(f"Union DEG summary: {summary_path}")


if __name__ == "__main__":
    main()
