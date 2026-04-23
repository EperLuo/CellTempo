# CellTempo

**Autoregressive forecasting of single-cell state transitions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-orange.svg)](https://pytorch.org/)
[![Dataset: scBaseTraj](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-scBaseTraj-blue)](https://huggingface.co/datasets/EperLuo/scBaseTraj)

CellTempo is a **temporal single-cell foundation model** that learns cellular
dynamics as a generative process and forecasts long-range cell-state transition
trajectories from snapshot measurements in a fully **autoregressive** manner.

- It first compresses each single-cell transcriptome into a compact sequence of
  **discrete cell codes** with a Vector-Quantized VAE (CellTempo-VQVAE).
- It then models cellular temporal progression with a **decoder-only
  Transformer** (CellTempo-Backbone) that autoregressively predicts ordered
  sequences of cell codes, where each sequence corresponds to a biologically
  grounded multi-step trajectory.

With this design, CellTempo can:

- **Forecast cell-state evolution** starting from any individual snapshot cell.
- **Reconstruct cellular potential landscapes** that recover preferred
  directions and tendencies of cell-state progression (e.g. hematopoietic
  differentiation hierarchies).
- **Predict perturbation responses** — both genetic (perturbation of
  lineage-associated gene modules) and chemical (e.g. ATRA in hematopoietic
  stem cells, anti-cancer drugs in breast/colorectal cancer cell lines) — and
  capture both immediate and delayed drug responses.

---

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [Environment & Installation](#environment--installation)
- [Data Preparation](#data-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Training](#training)
  - [Stage 1 — CellTempo-VQVAE](#stage-1--celltempo-vqvae)
  - [Stage 2 — CellTempo-Backbone](#stage-2--celltempo-backbone)
  - [Fine-tuning on Chemical Perturbations (Tahoe-100M)](#fine-tuning-on-chemical-perturbations-tahoe-100m)
- [Inference & Generation](#inference--generation)
- [Evaluation & Reproducing Main Results](#evaluation--reproducing-main-results)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)
- [License](#license)

---

## Overview

Snapshot single-cell RNA-seq data captures only one moment of a cell's life.
CellTempo reframes trajectory inference as **autoregressive sequence modeling**:
it treats a cell's transcriptome as a short sequence of discrete tokens, and
treats a cellular trajectory as a longer sequence of such token blocks. A
large-scale pre-training dataset — **scBaseTraj** — provides biologically
grounded multi-step supervision so that the model can generalize trajectory
generation to arbitrary unseen snapshot data.

**scBaseTraj** (hosted at
[EperLuo/scBaseTraj](https://huggingface.co/datasets/EperLuo/scBaseTraj))
integrates **RNA velocity**, **pseudotime**, and **inferred transition
probabilities** to compose more than **48 million trajectories spanning 71
tissues**, with each trajectory covering on average **9.3 consecutive cell
states**. Trajectories are indexed by **CytoTRACE2** developmental potential
stages: ~3/4 of them stay within a single stage (stable states), while the
rest span multiple stages (dynamic transitions).

## Model Architecture

CellTempo is a two-stage foundation model.

### Stage 1 — CellTempo-VQVAE (cell tokenizer)
- Compresses a single-cell transcriptome (default: **18,791 HVGs**, see
  [`src/utils/OS_scRNA_gene_index.18791.tsv`](src/utils/OS_scRNA_gene_index.18791.tsv))
  into a small set of **discrete codes** via vector quantization.
- Uses a **negative-binomial (NB) reconstruction loss** to respect the count
  nature of scRNA-seq data (configurable Poisson / NB / ZINB through
  `--vae_loss`).
- Implementation: [`src/model/CellTempo_VQVAE/`](src/model/CellTempo_VQVAE).
- Default codebook size: **512 codes** (`vocab_size = 512`).

### Stage 2 — CellTempo-Backbone (trajectory generator)
- Decoder-only Transformer with **RMSNorm**, GELU feed-forward, learned
  absolute positional embeddings, and **dedicated cell-position embeddings**
  (`cell_pos_num`) that mark the position of each cell within a trajectory.
- Autoregressively predicts the next cell's codes given the full history of
  previous cells in the trajectory, enabling arbitrary-horizon generation.
- Default size: **n_layer = 24, n_head = 20, n_embd = 1120** (~300M params).
- Implementation: [`src/model/CellTempo_backbone.py`](src/model/CellTempo_backbone.py).

### Token vocabulary
The full vocabulary (see
[`data/mix_meta_info_vq_traj.json`](data/mix_meta_info_vq_traj.json)) contains:

| Token group | Count | Meaning |
|---|---|---|
| VQ code tokens `0 … 511` | 512 | Discrete codes from CellTempo-VQVAE |
| Task tokens | 5 | `velo_g0`, `velo_g1`, `perturb`, `knockD`, `knockU` |
| Trajectory-position tokens | 20 | `traj_0 … traj_19` |
| Special tokens | 3 + 500 | `<S>`, `<E>`, `<U>` and reserved `<SPToken1>…<SPToken500>` |

## Repository Structure

```
CellTempo/
├── configs/                          # YAML configs for pretraining / fine-tuning / inference
│   ├── celltempo_scbasetraj_pretrain.yaml   # Backbone pretraining on scBaseTraj
│   ├── tahoe100m_finetune.yaml              # Fine-tuning on Tahoe-100M perturbation data
│   ├── generate_traj_scBasetraj_testset.yaml# Inference on scBaseTraj test set
│   └── generate_traj_h5ad_file.yaml         # Inference on a user-provided h5ad
│
├── data/                             # Small metadata files shipped with the repo
│   ├── mix_meta_info_vq_traj.json    # Vocabulary (gene_set / task_set / sp_token_set / token_set)
│   ├── sample_metadata.parquet       # Per-sample metadata (tissue, study, …)
│   └── size_factor.pkl               # Precomputed size factors for normalization
│
├── notebooks/                        # Reproduce the main results in the paper
│   ├── evaluation_scbasetraj_testset.ipynb  # scBaseTraj test-set benchmark
│   ├── evaluation_h5ad_data.ipynb           # Evaluate on external h5ad datasets
│   └── evaluation_landscape.ipynb           # Hematopoiesis landscape + perturbation
│
├── scripts/                          # Shell entry points
│   ├── trainer_celltempo_vqvae.sh
│   ├── trainer_celltempo_backbone.sh
│   └── generator_traj.sh
│
├── src/
│   ├── trainer_celltempo_vqvae.py    # Stage-1 training (accelerate)
│   ├── trainer_celltempo_backbone.py # Stage-2 training (torchrun / HF Trainer)
│   ├── generate_traj.py              # Inference entry point
│   ├── inference.py                  # Parallel trajectory generation
│   ├── experiments_log.json          # Reference experiment log
│   ├── model/
│   │   ├── CellTempo_backbone.py     # Transformer decoder
│   │   └── CellTempo_VQVAE/          # VQ-VAE encoder/decoder/quantizer
│   └── utils/
│       ├── tokenizer.py              # mixMulanTokenizer (maps codes ↔ tokens)
│       ├── dataset.py                # Trajectory / perturbation datasets & collate fns
│       ├── train_utils.py            # Dataset initialization helpers
│       ├── distribution.py           # NB / ZINB distributions for VAE loss
│       ├── screen.py                 # Perturbation-screen utilities
│       ├── utils_metrics.py          # Evaluation metrics
│       ├── cfg.py                    # Config helpers
│       └── OS_scRNA_gene_index.18791.tsv  # Global 18,791-gene reference
│
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

> The directories `outputs/` (checkpoints / generated trajectories) and the
> raw scBaseTraj data folder are not shipped with the repo; they are created
> on first run.

## Environment & Installation

Tested with **Python 3.10**, **CUDA 11.8**, **PyTorch 2.0.1**, on Linux.

```bash
conda create -n celltempo python=3.10 -y
conda activate celltempo

pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

Additional dependency — **CytoTRACE2** (used for cellular potential scoring
during evaluation and landscape reconstruction):

```bash
git clone https://github.com/digitalcytometry/cytotrace2
cd cytotrace2/cytotrace2_python
pip install -e .
```

> ⚠️ **Known issue.** Some versions of `scib` published on PyPI fail to build
> certain Cython modules. If you hit installation errors, install from source:
> <https://github.com/theislab/scib>.

## Data Preparation

1. **scBaseTraj (pre-training data).** Download from Hugging Face:

   ```bash
   huggingface-cli download EperLuo/scBaseTraj --repo-type dataset \
       --local-dir data/scBaseTraj
   ```

   Then point `data_folders` / `dataset_names` in the YAML config to this
   directory. The dataset ships with:
   - Per-cell expression data (raw counts on the 18,791-gene reference).
   - Per-trajectory metadata: indices of ordered cell states, pseudotime,
     RNA velocity, and CytoTRACE2 stage.

2. **Repository-local metadata.** The small bookkeeping files in `data/`
   (vocabulary, size factors, sample metadata) are already included so that
   `generate_traj.py` and the training scripts run out-of-the-box.

3. **Custom h5ad input for inference.** Any `.h5ad` file whose `var_names`
   contain the 18,791 reference genes (extra genes are ignored, missing ones
   are zero-padded) can be used with `--infer_type trajectory_h5ad`.

## Pretrained Checkpoints

Pretrained VQVAE and backbone checkpoints used in the paper can be found in Huggingface: [EperLuo/scBaseTraj](https://huggingface.co/datasets/EperLuo/scBaseTraj). Once downloaded, update the following
fields in the inference YAML:

```yaml
vq_vae_path: /path/to/vqvae_checkpoint/vqmodel
ckpt_path:   /path/to/backbone_checkpoint
```

## Training

> **Before running any script**, edit the absolute paths in the YAML files and
> in the shell scripts under `scripts/` so they match your environment
> (`output_dir`, `data_folders`, `vq_vae_path`, `ckpt_path`, W&B key, etc.).

### Stage 1 — CellTempo-VQVAE

Train the single-cell tokenizer with `accelerate`:

```bash
bash scripts/trainer_celltempo_vqvae.sh
```

Key flags (see [`scripts/trainer_celltempo_vqvae.sh`](scripts/trainer_celltempo_vqvae.sh)):

| Flag | Default | Description |
|---|---|---|
| `--num_gene` | `18791` | Size of the input gene panel |
| `--resolution` | `128` | Latent channel size |
| `--vae_loss` | `nb` | Reconstruction distribution (`nb` / `zinb` / `poisson` / `mse`) |
| `--train_batch_size` | `64` | Per-GPU batch size |
| `--max_train_steps` | `1,200,000` | Total optimizer steps |
| `--learning_rate` | `1e-4` | AdamW learning rate (linear schedule) |
| `--vq` |  | Enable vector quantization (disable for a vanilla VAE baseline) |
| `--resume_from_checkpoint` | `latest` | Auto-resume from the newest checkpoint in `output_dir` |

The final VQ model is saved under
`<output_dir>/checkpoint-<step>/vqmodel` and is consumed by Stage 2.

### Stage 2 — CellTempo-Backbone

Train the autoregressive Transformer on scBaseTraj using `torchrun` (multi-node
ready — the provided launcher reads `MLP_WORKER_*` environment variables used
on many clusters):

```bash
bash scripts/trainer_celltempo_backbone.sh
```

This loads [`configs/celltempo_scbasetraj_pretrain.yaml`](configs/celltempo_scbasetraj_pretrain.yaml)

The Stage-1 VQVAE path must be set via `vq_vae_path` before launching.

## Inference & Generation

All inference tasks are driven by a single entry point,
[`src/generate_traj.py`](src/generate_traj.py), which supports three modes:

| `--infer_type` | What it does |
|---|---|
| `trajectory_scbasetraj`    | Generate trajectories starting from the **scBaseTraj test set** (in-distribution benchmark). |
| `trajectory_h5ad`          | Generate trajectories starting from any **user-provided `.h5ad`** snapshot. |
| `trajectory_perturb_h5ad`  | Given a trajectory data, **perturb** one or more intermediate cells, and let the model continue — used to build counterfactual / landscape experiments. |

A minimal example (see [`scripts/generator_traj.sh`](scripts/generator_traj.sh)):

For the `trajectory_perturb_h5ad` mode, you must first construct trajectories
and specify which intermediate cell(s) to perturb.

## Evaluation

The three notebooks under `notebooks/` reproduce the quantitative benchmarks
and the main biological analyses reported in the paper.

| Notebook | Purpose |
|---|---|
| `evaluation_scbasetraj_testset.ipynb` | Held-out benchmark on scBaseTraj: predicted vs. ground-truth sequences, per-stage accuracy, CytoTRACE2 progression. |
| `evaluation_h5ad_data.ipynb` | External validation on independent snapshot datasets (from scBasecount. Can also be any datasets with spliced/unspliced information). |
| `evaluation_landscape.ipynb` | Build the **cellular potential landscape** of human hematopoiesis with CellTempo, and run lineage-gene perturbations. |

Each notebook assumes that inference has already been run with the matching
`--infer_type`. Re-run the cells top-to-bottom; intermediate artefacts are
written next to the notebook.

TODO: Long-term trajectory prediction for drug perturbation.

## Configuration Reference

| Config file | Role |
|---|---|
| `configs/celltempo_scbasetraj_pretrain.yaml` | Backbone pretraining on scBaseTraj |
| `configs/generate_traj_scBasetraj_testset.yaml` | Inference on the scBaseTraj test split |
| `configs/generate_traj_h5ad_file.yaml` | Inference on a user-provided `.h5ad` |

All paths in the YAML files are currently set for the authors' cluster; please update them before running.

## Citation

If you find CellTempo or the scBaseTraj dataset useful, please cite:

```bibtex
@article{luo2026celltempo,
  title   = {CellTempo: Autoregressive forecasting of single-cell state transitions},
  author  = {Luo, Erpai and others},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.08.704720},
  url     = {https://www.biorxiv.org/content/10.64898/2026.02.08.704720v1}
}
```

## License

This project is released under the [MIT License](LICENSE).

