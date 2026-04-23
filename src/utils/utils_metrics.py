import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
import scib
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from torch.autograd import Variable

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr,entropy
from scipy import stats
from statsmodels.stats.multitest import multipletests

import os, re, glob, math, time, json
from datasets import load_dataset, load_from_disk, concatenate_datasets

import pandas as pd
from celltypist import models, annotate
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

import networkx as nx
from scipy.sparse import issparse
from scipy.stats import spearmanr

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---- optional: pygam (skipped automatically if not installed) ----
_HAVE_GAM = False
try:
    from pygam import LinearGAM, s
    _HAVE_GAM = True
except Exception:
    _HAVE_GAM = False

from collections import defaultdict

def group_by_step_unique_first(trajectories):
    """
    For each cell_id, keep only the occurrence with the smallest time_step
    (trajectory position index) across all trajectories, then group by the
    original time_step.

    trajectories: List[List[cell_id]]

    Returns:
      grouped_nonempty: [[cell_ids at step s0], [cell_ids at step s1], ...]
      steps_sorted:     [s0, s1, ...]  corresponding original step indices
      first_pos:        {cell_id: (traj_idx, step_idx)}  for debugging / reuse
    """
    first_pos = {}  # cell_id -> (traj_idx, step_idx)

    for traj_idx, traj in enumerate(trajectories):
        for step_idx, cell_id in enumerate(traj):
            if cell_id not in first_pos or step_idx < first_pos[cell_id][1]:
                first_pos[cell_id] = (traj_idx, step_idx)

    # group by step_idx
    step_to_cells = defaultdict(list)
    for cell_id, (_t, step_idx) in first_pos.items():
        step_to_cells[step_idx].append(cell_id)

    steps_sorted = sorted(step_to_cells.keys())
    grouped_nonempty = [step_to_cells[s] for s in steps_sorted]
    return grouped_nonempty, steps_sorted, first_pos


def group_by_step_unique_last(trajectories):
    """
    For each cell_id, keep only the occurrence with the largest time_step
    across all trajectories, then group by the original time_step.

    Returns:
      grouped_nonempty, steps_sorted, last_pos — same structure as above
    """
    last_pos = {}  # cell_id -> (traj_idx, step_idx)

    for traj_idx, traj in enumerate(trajectories):
        for step_idx, cell_id in enumerate(traj):
            # keep only the occurrence with the largest step_idx
            if cell_id not in last_pos or step_idx > last_pos[cell_id][1]:
                last_pos[cell_id] = (traj_idx, step_idx)

    step_to_cells = defaultdict(list)
    for cell_id, (_t, step_idx) in last_pos.items():
        step_to_cells[step_idx].append(cell_id)

    steps_sorted = sorted(step_to_cells.keys())
    grouped_nonempty = [step_to_cells[s] for s in steps_sorted]
    return grouped_nonempty, steps_sorted, last_pos

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

# used to find the next cell
def get_next_cell_velo(idx_num, global_dataset):

    if idx_num is None:
        print(f'idx num {idx_num} is None.')
        return None, None
    cell_data = global_dataset[idx_num]
    gene_names = cell_data['expressed_genes']#[:1024]
    expr_values = cell_data['expressed_values']#[:1024]
    # expr_values = torch.clamp(torch.tensor(expr_values), min=1e-5, max=30).tolist()

    return gene_names, expr_values

# align generated genes to the reference gene panel
def align_expression_to_reference(gene_lists, expression_lists, reference_genes):
    ref_index = {gene: idx for idx, gene in enumerate(reference_genes)}
    aligned_results = []

    for genes, exprs in zip(gene_lists, expression_lists):
        aligned = [0.0] * len(reference_genes)
        for g, e in zip(genes, exprs):
            if g in ref_index:
                aligned[ref_index[g]] = e
        aligned_results.append(aligned)
    
    return np.stack(aligned_results)



def knn_preservation(X_ref, X_code, k=20, metric_ref='euclidean', metric_code='hamming'):
    """Compute the kNN neighbour preservation rate."""
    nbr_ref = NearestNeighbors(n_neighbors=k, metric=metric_ref).fit(X_ref)
    nbr_code = NearestNeighbors(n_neighbors=k, metric=metric_code).fit(X_code)
    idx_ref = nbr_ref.kneighbors(return_distance=False)
    idx_code = nbr_code.kneighbors(return_distance=False)
    overlap = [len(set(a).intersection(b)) / k for a, b in zip(idx_ref, idx_code)]
    return np.mean(overlap)


def distance_correlation(X_ref, X_code, metric_ref='euclidean', metric_code='hamming', subsample=2000):
    """Compute pairwise distance correlation (global structure preservation)."""
    # subsample to avoid excessive memory usage
    if X_ref.shape[0] > subsample:
        idx = np.random.choice(X_ref.shape[0], subsample, replace=False)
        X_ref = X_ref[idx]
        X_code = X_code[idx]
    D1 = squareform(pdist(X_ref, metric_ref))
    D2 = squareform(pdist(X_code, metric_code))
    r, _ = pearsonr(D1.ravel(), D2.ravel())
    return r


def cluster_consistency(X_ref, X_code, n_clusters=10, metric_code='hamming'):
    """Compute clustering consistency (ARI/NMI)."""
    kmeans_ref = KMeans(n_clusters=n_clusters, random_state=0).fit(X_ref)
    # when clustering in code space, use Hamming distance (one-hot or integer representation)
    kmeans_code = KMeans(n_clusters=n_clusters, random_state=0).fit(X_code)
    ari = adjusted_rand_score(kmeans_ref.labels_, kmeans_code.labels_)
    nmi = normalized_mutual_info_score(kmeans_ref.labels_, kmeans_code.labels_)
    return ari, nmi


def evaluate_all(X_orig, X_code, labels=None, n_clusters=10, k=20):
    print("==== Evaluating VQ-VAE code structure ====")
    results = {}

    # kNN preservation rate
    knn_rate = knn_preservation(X_orig, X_code, k=k)
    results["kNN_preservation"] = knn_rate
    print(f"[1] kNN preservation rate: {knn_rate:.4f}")

    # distance correlation
    dist_corr = distance_correlation(X_orig, X_code)
    results["distance_correlation"] = dist_corr
    print(f"[2] Distance correlation: {dist_corr:.4f}")

    # clustering consistency
    ari, nmi = cluster_consistency(X_orig, X_code, n_clusters=n_clusters)
    results["ARI"] = ari
    results["NMI"] = nmi
    print(f"[3] Cluster consistency - ARI: {ari:.4f}, NMI: {nmi:.4f}")

    # Trustworthiness
    trust = trustworthiness(X_orig, X_code, n_neighbors=k)
    results["trustworthiness"] = trust
    print(f"[4] Trustworthiness: {trust:.4f}")

    # if labels are provided, also compute label-to-cluster consistency
    if labels is not None:
        labels = np.array(labels)
        km_code = KMeans(n_clusters=len(np.unique(labels)), random_state=0).fit(X_code)
        ari_lbl = adjusted_rand_score(labels, km_code.labels_)
        nmi_lbl = normalized_mutual_info_score(labels, km_code.labels_)
        results["ARI_label_vs_code"] = ari_lbl
        results["NMI_label_vs_code"] = nmi_lbl
        print(f"[5] Label vs Code clusters - ARI: {ari_lbl:.4f}, NMI: {nmi_lbl:.4f}")

    print("==== Done ====")
    return results


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Convert source and target domain data into a kernel matrix K.
    Params:
	    source: source domain data (n * len(x))
	    target: target domain data (m * len(y))
	    kernel_mul:
	    kernel_num: number of different Gaussian kernels
	    fix_sigma: sigma values for the Gaussian kernels
	Return:
		sum(kernel_val): sum of multiple kernel matrices
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# total number of rows; source and target are typically the same size
    total = torch.cat([source, target], dim=0)# concatenate source and target along columns
    # tile total (n+m) times
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # expand each row of total to (n+m) rows
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # pairwise L2 distances; entry (i,j) = L2 distance between row i and row j of total (0 when i==j)
    
    batch_size = 200
    num_window = int(total0.shape[0]/batch_size)+1
    L2_dis = []
    for i in tqdm(range(num_window)):
        diff = (total0[i*batch_size:(i+1)*batch_size]-total1[i*batch_size:(i+1)*batch_size])#.cuda()
        diff.square_()
        L2_dis.append(diff.sum(2).cpu())
    L2_distance = torch.concatenate(L2_dis,dim=0)

    # L2_distance = ((total0-total1)**2).sum(2) 

    # set the bandwidth (sigma) for the Gaussian kernel
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # take kernel_num bandwidth values centred on fix_sigma with multiplier kernel_mul (e.g. [0.25,0.5,1,2,4] when fix_sigma=1)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # Gaussian kernel function
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # produce the final kernel matrix
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Compute the MMD distance between source and target domain data.
    Params:
	    source: source domain data (n * len(x))
	    target: target domain data (m * len(y))
	    kernel_mul:
	    kernel_num: number of different Gaussian kernels
	    fix_sigma: sigma values for the Gaussian kernels
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])# source and target are typically the same batch size
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # split the kernel matrix into 4 blocks following equation (3)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss# the L matrix is typically omitted because n==m

def MMD(adata):
    real = adata[adata.obs['batch']=='true_Cell'].obsm['X_pca']
    gen = adata[adata.obs['batch']=='gen_Cell'].obsm['X_pca']
    X = torch.Tensor(real)
    Y = torch.Tensor(gen)
    X,Y = Variable(X), Variable(Y)
    return mmd_rbf(X,Y)


def LISI(adata):
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    lisi = scib.me.ilisi_graph(adata, batch_key="batch", type_="knn")
    return lisi

def random_forest(adata, return_roc = False):
    real = adata[adata.obs['batch']=='true_Cell'].obsm['X_pca']
    sim = adata[adata.obs['batch']=='gen_Cell'].obsm['X_pca']

    data = np.concatenate((real,sim),axis=0)
    label = np.concatenate((np.ones((real.shape[0])),np.zeros((sim.shape[0]))))

    # split into training and validation sets
    X_train,X_val,y_train,y_val = train_test_split(data, label,
                                                test_size = 0.25,random_state = 1)

    # classify with Random Forest
    rfc1 = RandomForestClassifier(n_estimators = 1000, # number of trees
                                max_depth= 5,       # max depth of each tree
                                oob_score=True,
                                class_weight = "balanced",
                                random_state=1)
    rfc1.fit(X_train,y_train)

    # ROC curve on the validation set
    pre_y = rfc1.predict_proba(X_val)[:, 1]
    fpr_Nb, tpr_Nb, _ = roc_curve(y_val, pre_y)
    aucval = auc(fpr_Nb, tpr_Nb)    # compute AUC
    if return_roc:
        return aucval, fpr_Nb, tpr_Nb
    return aucval


def calculate_mse(array):
    # get the size of the first dimension
    n, h, w, d = array.shape
    
    # initialise result array
    mse_values = np.zeros((n - 1, h))
    
    # compute MSE between each consecutive pair of tensors
    for i in range(1, n):
        # compute per-position MSE between adjacent tensors
        mse = np.mean((array[i] - array[i - 1]) ** 2, axis=(1, 2))
        mse_values[i - 1] = mse
    
    return mse_values


def map_adata_to_reference_genes(adata, ref_genes):
    """
    Map genes in adata to the reference gene list ref_genes.
    - Missing genes are zero-padded.
    - Extra genes are dropped.
    - Output adata.var follows the order of ref_genes.

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


from scipy.stats import pearsonr,entropy
# ---------------------- CellTypist end-to-end trajectory composition analysis ----------------------

# ========== 1) Configuration ==========
# choose a model (can be 'Immune_All_High.pkl' / 'Immune_All_Low.pkl' / 'Pan_fetal_cell.pkl', etc.)
CELLTYPIST_MODEL = 'Immune_All_Low.pkl'   # change according to your data domain

# ========== 2) Utility functions ==========
def ensure_step_col(adata, prefer=('step', 'time_step')):
    for k in prefer:
        if k in adata.obs.columns:
            return k
    raise KeyError(f"Cannot find step column; please provide one of {prefer} in adata.obs.")

def run_celltypist(adata, model_name='Immune_All_Low.pkl', majority_voting=True, label_col='cell_type'):
    """
    Annotate with CellTypist; results written to adata.obs[label_col].
    model_name can be a model filename or a full path.
    """
    # download / ensure the model is available
    models.download_models(model=[model_name], force_update=False)
    # load model
    model = models.Model.load(model=model_name)
    # annotate
    result = annotate(adata, model=model, majority_voting=majority_voting)
    if hasattr(result, 'predicted_labels_majority_voting'):
        adata.obs[label_col] = result.predicted_labels['majority_voting']
    else:
        adata.obs[label_col] = result.predicted_labels['predicted_labels']
    return adata

def comp_by_step(adata, step_col, type_col='cell_type'):
    """
    Compute cell-type composition per step (rows=step, columns=cell_type, values=proportion).
    """
    comp = (
        adata.obs
        .groupby(step_col)[type_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .sort_index()
    )
    return comp

def align_columns(a: pd.DataFrame, b: pd.DataFrame):
    """
    Align columns of two composition DataFrames (union of cell types), fill missing with 0, sort by column name.
    """
    cols = sorted(set(a.columns) | set(b.columns))
    return a.reindex(columns=cols, fill_value=0.0), b.reindex(columns=cols, fill_value=0.0)

def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """
    JSD(p||q). p and q must be probability distributions (sum to 1); numerical stability handled internally.
    """
    p = np.clip(p, 1e-12, 1.0); p = p / p.sum()
    q = np.clip(q, 1e-12, 1.0); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compare_composition(comp_real: pd.DataFrame, comp_gen: pd.DataFrame):
    """
    Returns:
      - jsd: JSD per step
      - diff: composition difference (comp_gen - comp_real) for heatmaps
    """
    # compare only steps present in both datasets (can be changed to union with zero-fill)
    steps = comp_real.index.intersection(comp_gen.index)
    comp_r = comp_real.loc[steps]
    comp_g = comp_gen.loc[steps]
    # JSD
    jsd = pd.Series(
        {s: jensen_shannon(comp_r.loc[s].values, comp_g.loc[s].values) for s in steps},
        name='JSD'
    )
    # difference matrix (gen - real)
    diff = (comp_g - comp_r).rename_axis('step')
    return jsd.sort_index(), diff.sort_index()

def stacked_bar(
    df: pd.DataFrame,
    title: str,
    save_path: str,
    color_map: dict,
    min_frac: float = 0.05,
    other_name: str = "Other",
):
    df_plot = df.copy()

    # --------- merge low-proportion columns ---------
    if min_frac is not None:
        col_max = df_plot.max(axis=0)
        major = col_max[col_max >= min_frac].index
        minor = col_max[col_max < min_frac].index

        if len(minor) > 0:
            df_major = df_plot[major].copy()
            df_major[other_name] = df_plot[minor].sum(axis=1)
            df_plot = df_major

    # (optional) sort by column mean
    df_plot = df_plot[df_plot.mean(axis=0).sort_values(ascending=False).index]

    # --------- key: assign fixed colours by column name ---------
    colors = [color_map[c] for c in df_plot.columns]

    # --------- plotting ---------
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(
        kind="bar",
        stacked=True,
        width=0.9,
        ax=ax,
        color=colors,
    )

    ax.set_ylabel("Proportion")
    ax.set_title(title)

    ax.legend(
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=5,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f"{save_path}/{title}.pdf")
    # plt.close()


def heatmap_diff(diff: pd.DataFrame, vcenter=0.0, vmax=None, title='Composition difference (gen - real)'):
    """
    Simple heatmap (positive values indicate higher cell-type proportion in generated vs. real data).
    """
    import matplotlib.colors as mcolors
    vmax = vmax or np.max(np.abs(diff.values))
    cmap = plt.get_cmap('bwr')  # blue-white-red; white = 0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=vcenter, vmax=vmax)

    plt.figure(figsize=(max(8, diff.shape[1]*0.4), max(8, diff.shape[0]*0.3)))
    plt.imshow(diff.values, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(label='gen - real')
    plt.xticks(range(diff.shape[1]), diff.columns, rotation=90)
    plt.yticks(range(diff.shape[0]), diff.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_celltypist_by_step(adata, step_col, model_name, label_col='cell_type'):
    """
    Run CellTypist separately for each step group on the same AnnData object,
    then write the predicted labels back to adata.obs[label_col].

    label_col is stored as object/string type to avoid pandas Categorical conflicts.
    """
    # pre-create an empty object column to avoid Categorical type incompatibility later
    if label_col not in adata.obs.columns:
        adata.obs[label_col] = pd.Series(index=adata.obs_names, dtype="object")
    else:
        # if already Categorical, convert to object first
        if pd.api.types.is_categorical_dtype(adata.obs[label_col].dtype):
            adata.obs[label_col] = adata.obs[label_col].astype("object")

    steps = adata.obs[step_col].unique()
    try:
        steps = sorted(steps)
    except Exception:
        pass

    for s in steps:
        mask = (adata.obs[step_col] == s)
        if mask.sum() == 0:
            continue
        print(f"[CellTypist] Annotating step={s} (n={mask.sum()}) ...")

        sub = adata[mask].copy()
        run_celltypist(sub, model_name=model_name, label_col=label_col)

        # convert subset predictions to str before writing back, to avoid carrying Categorical type
        adata.obs.loc[sub.obs_names, label_col] = sub.obs[label_col].astype(str).values
# ========== 3) Main pipeline (pass your AnnData here) ==========
def celltype_composition_pipeline(adata_real, adata_gen,
                                  model_name=CELLTYPIST_MODEL,
                                  step_candidates=('step', 'time_step'),
                                  type_col='cell_type',
                                  export_prefix=None):
    """
    All-in-one pipeline:
      - Annotate cell types in real/generated data
      - Compute per-step type composition
      - Compute per-step JSD and composition difference
      - Plot (stacked bar, difference heatmap)
      - (Optional) export CSV files
    """
    sc.pp.normalize_total(adata_real, 1e4)
    sc.pp.log1p(adata_real)
    sc.pp.normalize_total(adata_gen, 1e4)
    sc.pp.log1p(adata_gen)


    # select step column
    step_r = ensure_step_col(adata_real, step_candidates)
    step_g = ensure_step_col(adata_gen,  step_candidates)
    if step_r != step_g:
        print(f"[WARN] Real and generated datasets have different step column names: real={step_r}, gen={step_g}. Comparing over intersecting steps.")

    # annotate
    print("[CellTypist] Annotating REAL ...")
    # run_celltypist(adata_real, model_name=model_name, label_col=type_col)
    run_celltypist_by_step(adata_real, step_col=step_r,
                           model_name=model_name, label_col=type_col)
    print("[CellTypist] Annotating GEN  ...")
    # run_celltypist(adata_gen,  model_name=model_name, label_col=type_col)
    run_celltypist_by_step(adata_gen,  step_col=step_g,
                           model_name=model_name, label_col=type_col)

    # composition
    comp_real = comp_by_step(adata_real, step_col=step_r, type_col=type_col)
    comp_gen  = comp_by_step(adata_gen,  step_col=step_g, type_col=type_col)
    comp_real, comp_gen = align_columns(comp_real, comp_gen)

    # comparison
    jsd, diff = compare_composition(comp_real, comp_gen)

    # # print top differing steps
    # print("\n[Summary] Top-10 steps by JSD (higher = more different):")
    # print(jsd.sort_values(ascending=False).head(10))
    from matplotlib import cm

    def make_color_map(categories, cmap_name="tab20"):
        cmap = cm.get_cmap(cmap_name, len(categories))
        return {cat: cmap(i) for i, cat in enumerate(categories)}

    # --------- merge low-proportion columns ---------
    min_frac = 0.05
    df_plot = comp_real
    col_max = df_plot.max(axis=0)
    major = col_max[col_max >= min_frac].index
    minor = col_max[col_max < min_frac].index

    if len(minor) > 0:
        df_major = df_plot[major].copy()
        df_major['Other'] = df_plot[minor].sum(axis=1)
        df_plot = df_major
    comp_real = df_plot

    df_plot = comp_gen
    col_max = df_plot.max(axis=0)
    major = col_max[col_max >= min_frac].index
    minor = col_max[col_max < min_frac].index

    if len(minor) > 0:
        df_major = df_plot[major].copy()
        df_major['Other'] = df_plot[minor].sum(axis=1)
        df_plot = df_major
    comp_gen = df_plot

    all_categories = np.unique(np.concatenate([comp_real.columns.values,comp_gen.columns.values]))
        
    COLOR_MAP = make_color_map(all_categories)

    # plotting
    stacked_bar(
        comp_real,
        title='Real: cell type composition per step',
        save_path='/'.join(export_prefix.split('/')[:-1]),
        color_map=COLOR_MAP,
    )

    stacked_bar(
        comp_gen,
        title='Generated: cell type composition per step',
        save_path='/'.join(export_prefix.split('/')[:-1]),
        color_map=COLOR_MAP,
    )
    # stacked_bar(comp_real, title='Real: cell type composition per step', save_path='/'.join(export_prefix.split('/')[:-1]))
    # stacked_bar(comp_gen,  title='Generated: cell type composition per step', save_path='/'.join(export_prefix.split('/')[:-1]))
    heatmap_diff(diff, title='Composition difference (gen - real)')

    # optional export
    if export_prefix:
        comp_real.to_csv(f"{export_prefix}_composition_real.csv")
        comp_gen.to_csv(f"{export_prefix}_composition_gen.csv")
        jsd.to_csv(f"{export_prefix}_JSD_by_step.csv")
        diff.to_csv(f"{export_prefix}_composition_diff_gen_minus_real.csv")
        print(f"[Export] Exported to prefix {export_prefix}_*.csv")

    # return results for further custom analysis
    return {
        "comp_real": comp_real,
        "comp_gen": comp_gen,
        "jsd_by_step": jsd,
        "diff_matrix": diff
    }



# ----------------- utils -----------------
def _get_X_dense(adata, layer=None):
    X = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)

def _time_to_numeric_and_levels(ser):
    """Convert time_step to numeric codes and return ordered levels (for aligning time points between two adata objects)."""
    s = ser.copy()
    if pd.api.types.is_numeric_dtype(s):
        codes = s.values.astype(float)
        levels = pd.Index(np.sort(pd.unique(s.values)))
    else:
        # if not pd.api.types.is_categorical_dtype(s):
        s = pd.Categorical(s, ordered=True)
        # elif not s.cat.ordered:
        #     s = s.as_ordered()
        codes = s.codes.astype(float)
        levels = s.categories
    return codes, pd.Index(levels)

def _align_time_levels(adata_r, adata_g, time_key):
    """Ensure both adata objects share the same ordered time_step levels; return intersect_levels."""
    tr = adata_r.obs[time_key]
    tg = adata_g.obs[time_key]
    # convert to ordered category and align
    if not pd.api.types.is_categorical_dtype(tr):
        tr = pd.Categorical(tr, ordered=True)
    if not pd.api.types.is_categorical_dtype(tg):
        tg = pd.Categorical(tg, ordered=True)
    # intersect levels; real set ordering takes priority
    levels = [x for x in tr.categories if x in set(tg.categories)]
    if len(levels) < 2:
        raise ValueError(f"Intersection of time steps between the two datasets is <2; cannot compare: {levels}")
    adata_r.obs[time_key] = pd.Categorical(adata_r.obs[time_key], categories=levels, ordered=True)
    adata_g.obs[time_key] = pd.Categorical(adata_g.obs[time_key], categories=levels, ordered=True)
    return pd.Index(levels)

def _group_stats_by_time(X, codes):
    """Compute per-time-point mean and SEM given discrete time codes; returns dict: level -> (mean_vec, sem_vec, n)."""
    out = {}
    uniq = np.unique(codes[~np.isnan(codes)])
    for u in uniq:
        idx = (codes == u)
        if idx.sum() == 0:
            continue
        Y = X[idx, :]
        mean = np.nanmean(Y, axis=0)
        sem = stats.sem(Y, axis=0, nan_policy="omit") if idx.sum() > 1 else np.zeros(Y.shape[1])
        out[u] = (mean, sem, int(idx.sum()))
    return out

def _spearman_screen(X, t_codes):
    G = X.shape[1]
    rho = np.zeros(G); p = np.ones(G)
    for g in range(G):
        v = X[:, g]
        if np.allclose(v, v[0]):
            rho[g], p[g] = np.nan, 1.0
            continue
        r, pp = stats.spearmanr(v, t_codes, nan_policy="omit")
        rho[g] = np.nan if np.isnan(r) else r
        p[g] = 1.0 if np.isnan(pp) else pp
    return rho, p

def _kruskal_by_groups(X, t_codes):
    uniq = np.unique(t_codes[~np.isnan(t_codes)])
    G = X.shape[1]; p = np.ones(G); H = np.zeros(G)
    for g in range(G):
        groups = [X[t_codes==u, g] for u in uniq]
        valid = [arr for arr in groups if arr.size > 1]
        if len(valid) < 2:
            p[g], H[g] = 1.0, 0.0
            continue
        h, pp = stats.kruskal(*valid, nan_policy="omit")
        p[g], H[g] = (1.0 if np.isnan(pp) else pp), (0.0 if np.isnan(h) else h)
    return p, H

def _gam_test(X, t_codes, max_genes=3000, lam=0.6, splines=10):
    if not _HAVE_GAM:
        return None, None
    var = X.var(0)
    order = np.argsort(-var)[:min(max_genes, X.shape[1])]
    pvals = np.ones(X.shape[1]); r2 = np.zeros(X.shape[1])
    t01 = (t_codes - np.nanmin(t_codes)) / (np.nanmax(t_codes) - np.nanmin(t_codes) + 1e-9)
    Z = t01.reshape(-1,1)
    for g in order:
        y = X[:, g]
        if np.allclose(y, y[0]):
            pvals[g], r2[g] = 1.0, 0.0
            continue
        try:
            gam = LinearGAM(s(0, n_splines=splines)).gridsearch(Z, y, lam=lam)
            pvals[g] = gam.statistics_.get('p_values', [np.nan, 1.0])[1]
            r2[g] = gam.statistics_.get('pseudo_r2', 0.0)
        except Exception:
            pvals[g], r2[g] = 1.0, 0.0
    return pvals, r2

def _fisher(*p_lists):
    valids = [np.clip(np.asarray(p), 1e-300, 1.0) for p in p_lists if p is not None]
    if not valids: return None
    P = np.vstack(valids)
    stat = -2 * np.nansum(np.log(P), axis=0)
    df = 2 * P.shape[0]
    return 1 - stats.chi2.cdf(stat, df)

def _filter_uninformative_genes(X, min_detect_rate=0.05, min_groups=2, t_codes=None):
    """
    Filter genes detected at rate >= min_detect_rate in at least min_groups time groups.
    Returns indices of retained columns.
    """
    if t_codes is None:
        # global detection rate
        det = (X > 0).mean(0)
        return np.where(det >= min_detect_rate)[0]

    keep = []
    uniq = np.unique(t_codes[~np.isnan(t_codes)])
    for g in range(X.shape[1]):
        cnt = 0
        for u in uniq:
            arr = X[t_codes == u, g]
            if arr.size > 0 and (arr > 0).mean() >= min_detect_rate:
                cnt += 1
        if cnt >= min_groups:
            keep.append(g)
    return np.array(keep, dtype=int)

# ----------------- main pipeline -----------------
def timecourse_markers_real_then_compare(
    adata_real,
    adata_gen,
    time_key="time_step",
    top_markers=None,
    layer=None,
    use_spearman=True,
    use_kruskal=True,
    use_gam=True,
    gam_top=3000,
    fdr_method="fdr_bh",
    topn_markers=50,
):
    """
    1) Find time-course markers in the real dataset using time_key.
    2) Plot the temporal trend of these markers in both real and generated datasets.
    3) Compare the real vs. generated temporal trend correlation per marker
       (using the mean vector at shared time points).

    Returns:
      markers_df: markers identified from the real dataset (with statistics and FDR, sorted by significance)
      corr_df:    Pearson/Spearman correlation per marker (real mean vs. generated mean)
      plot_fn:    a plotting function plot_markers_grid(markers, ncols=3, figsize=(...))
    """
    # use only shared genes
    genes = adata_real.var_names.intersection(adata_gen.var_names)
    if len(genes) == 0:
        raise ValueError("No common genes between the two datasets")
    adata_r = adata_real[:, genes].copy()
    adata_g = adata_gen[:, genes].copy()

    # align time levels
    levels = _align_time_levels(adata_r, adata_g, time_key)

    Xr = _get_X_dense(adata_r, layer=layer)
    Xg = _get_X_dense(adata_g, layer=layer)

    # ------- group means by time (for plotting & correlation) -------
    # map time levels to codes (0..L-1) for sequence indexing
    level_to_code = {lv: i for i, lv in enumerate(levels)}
    r_codes_ord = np.array([level_to_code[lv] for lv in adata_r.obs[time_key]])
    g_codes_ord = np.array([level_to_code[lv] for lv in adata_g.obs[time_key]])

    r_stats = _group_stats_by_time(Xr, r_codes_ord)  # dict[code] -> (mean_vec, sem_vec, n)
    g_stats = _group_stats_by_time(Xg, g_codes_ord)

    # build per-marker mean time series (in levels order)
    def _mean_series(stats_dict, gi, L):
        arr = []
        for k in range(L):
            if k in stats_dict:
                arr.append(stats_dict[k][0][gi])
            else:
                arr.append(np.nan)
        return np.array(arr, dtype=float)

    L = len(levels)
    corr_rows = []
    for gene in top_markers:
        gi = int(np.where(genes == gene)[0][0])
        real_means = _mean_series(r_stats, gi, L)
        gen_means  = _mean_series(g_stats, gi, L)
        # correlate only at time points where both values are finite
        mask = np.isfinite(real_means) & np.isfinite(gen_means)
        if mask.sum() >= 2:
            r_pear, p_pear = stats.pearsonr(real_means[mask], gen_means[mask])
            r_spear, p_spear2 = stats.spearmanr(real_means[mask], gen_means[mask])
        else:
            r_pear, p_pear, r_spear, p_spear2 = np.nan, np.nan, np.nan, np.nan
        corr_rows.append({
            "gene": gene,
            "n_common_timepoints": int(mask.sum()),
            "pearson_r": r_pear, "pearson_p": p_pear,
            "spearman_r": r_spear, "spearman_p": p_spear2
        })
    corr_df = pd.DataFrame(corr_rows).set_index("gene")

    # ------- plotting helper -------
    def plot_markers_grid(markers, ncols=3, figsize=(12, 3.0), with_sem=True):
        """
        Plot temporal trends of markers: solid=real, dashed=generated; optional SEM shading.
        """
        markers = [m for m in markers if m in genes]
        if not markers:
            raise ValueError("No plottable genes")
        n = len(markers)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1]*nrows), sharex=True)
        axes = np.ravel(axes) if nrows*ncols>1 else [axes]

        x = np.arange(len(levels))
        for i, g in enumerate(markers):
            ax = axes[i]
            gi = int(np.where(genes == g)[0][0])

            # REAL
            y_r = _mean_series(r_stats, gi, L)
            ax.plot(x, y_r, "-", lw=2, label="REAL")
            if with_sem:
                sem_r = np.array([r_stats[k][1][gi] if k in r_stats else np.nan for k in range(L)])
                ax.fill_between(x, y_r-sem_r, y_r+sem_r, alpha=0.2)

            # GEN
            y_g = _mean_series(g_stats, gi, L)
            ax.plot(x, y_g, "--", lw=2, label="GEN")

            # include correlation in the title
            row = corr_df.loc[g] if g in corr_df.index else None
            subtitle = ""
            if row is not None and np.isfinite(row["spearman_r"]):
                subtitle = f" | ρ={row['spearman_r']:.2f}, r={row['pearson_r']:.2f}"
            ax.set_title(f"{g}{subtitle}")
            ax.set_xticks(x); ax.set_xticklabels(list(map(str, levels)), rotation=0)
            ax.grid(alpha=0.2)
        # hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        return fig

    return corr_df, plot_markers_grid


from scipy.sparse import csr_matrix

def largest_umap_component_mask(adata, n_neighbors=15, step_key="time_step", step0_value=0):
    """
    Build a graph on UMAP coordinates and return a boolean mask for the largest connected
    component; always keep all cells from step_0.
    """
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] does not exist; please run sc.tl.umap() first.")

    X = adata.obsm["X_umap"]
    G = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    G = 0.5 * (G + G.T)  # symmetrize

    # connected component partitioning
    _, labels = connected_components(G, directed=False)
    keep_label = np.bincount(labels).argmax()
    mask = labels == keep_label

    # unconditionally keep all step_0 cells
    if step_key in adata.obs:
        step0_mask = adata.obs[step_key].values == step0_value
        mask = np.logical_or(mask, step0_mask)

    return mask

def sample_trajectory(
    T: csr_matrix, start_cell: int, max_steps: int = 50, stop_cells=None,
    pseudotime=None, fate_prob=None, lineage=None,
    k: int = 30, temperature: float = 0.7, tol=1e-2,
    forbid_self=True, forbid_backtrack=True, forbid_hist_neighs=True,
    random_state=None
):
    rng = np.random.default_rng(random_state)
    n = T.shape[0]
    if stop_cells is None:
        stop_cells = set()
    elif not isinstance(stop_cells, set):
        stop_cells = set(stop_cells)

    # fate_prob handling
    if fate_prob is not None:
        if isinstance(fate_prob, dict):
            fp = fate_prob[lineage]
        else:
            fp = fate_prob if lineage is None else fate_prob[:, lineage]
    else:
        fp = None

    traj = [start_cell]
    prev = -1
    cur = start_cell
    hist_neighs = set()   # set of all historical neighbours

    for _ in range(max_steps - 1):
        row_start, row_end = T.indptr[cur], T.indptr[cur + 1]
        nbrs = T.indices[row_start:row_end]
        probs = T.data[row_start:row_end]
        if len(nbrs) == 0:
            break

        # build mask
        mask = np.ones_like(nbrs, dtype=bool)
        if forbid_self:
            mask &= (nbrs != cur)
        if forbid_backtrack and prev >= 0:
            mask &= (nbrs != prev)
        if forbid_hist_neighs and len(hist_neighs) > 0:
            mask &= ~np.isin(nbrs, list(hist_neighs))
        if pseudotime is not None:
            mask &= (pseudotime[nbrs] >= pseudotime[cur] - tol)
        if fp is not None:
            mask &= (fp[nbrs] >= fp[cur] - tol)

        nbrs, probs = nbrs[mask], probs[mask]
        if len(nbrs) == 0:
            break

        # top-k selection
        if len(nbrs) > k:
            topk = np.argpartition(probs, -k)[-k:]
            nbrs, probs = nbrs[topk], probs[topk]

        # temperature scaling + normalisation
        probs = np.maximum(probs, 0.0)
        if temperature is not None and temperature > 0:
            invT = 1.0 / temperature
            probs = probs ** invT
        s = probs.sum()
        if s <= 0:
            break
        probs = probs / s

        nxt = rng.choice(nbrs, p=probs)
        traj.append(nxt)

        # update prev / cur / hist_neighs
        prev = cur
        cur = nxt
        rs, re = T.indptr[prev], T.indptr[prev + 1]
        hist_neighs.update(T.indices[rs:re])  # add all neighbours of prev to the historical forbidden set

        if cur in stop_cells:
            break

    return traj

# ---------- Optional ElPiGraph import ----------
_have_elpi = False
try:
    import elpigraph
    _have_elpi = True
except Exception:
    _have_elpi = False

# ---------- Helpers ----------
def intersect_genes(adata_r, adata_g):
    common = np.intersect1d(adata_r.var_names, adata_g.var_names)
    if len(common) == 0:
        raise ValueError("No common genes between REAL and GENERATED datasets.")
    return adata_r[:, common].copy(), adata_g[:, common].copy()

def ensure_neighbors_umap(adata, n_pcs=50, neighbors=30, seed=42):
    """Minimal graph & embedding prerequisites for DPT/plots."""
    if "X_pca" not in adata.obsm:
        if "log1p" not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, adata.n_vars), subset=True, flavor="seurat_v3")
        sc.pp.scale(adata, zero_center=True, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack', random_state=seed)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=neighbors,
                        n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]),
                        random_state=seed)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=seed)
    if "leiden" not in adata.obs:
        sc.tl.leiden(adata, key_added="leiden", random_state=seed, resolution=1.0)

def ensure_pseudotime(adata, time_key, pt_key="dpt_pseudotime"):
    """Compute DPT pseudotime if missing; use earliest time_key as root hint."""
    if pt_key in adata.obs:
        return
    ensure_neighbors_umap(adata)
    # try pick a root from the minimum time step
    iroot = None
    if time_key in adata.obs:
        ts = adata.obs[time_key]
        if not pd.api.types.is_numeric_dtype(ts):
            ts_num = pd.Categorical(ts, ordered=True).codes
        else:
            ts_num = ts.values
        min_t = np.nanmin(ts_num)
        idx = np.where(ts_num == min_t)[0]
        if len(idx) > 0:
            adata.uns["iroot"] = int(idx[0])
    sc.tl.dpt(adata, n_dcs=10)  # writes adata.obs['dpt_pseudotime']
    if pt_key not in adata.obs:
        raise RuntimeError(f"Failed to compute {pt_key}; check graph connectivity.")

def basic_preprocess(adata, n_pcs=50, neighbors=30, seed=42):
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, adata.n_vars), subset=True, flavor="seurat_v3")
        sc.pp.scale(adata, zero_center=True, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack', random_state=seed)
        sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=n_pcs, random_state=seed)
        sc.tl.umap(adata, random_state=seed)
        sc.tl.leiden(adata, key_added="leiden", random_state=seed, resolution=1.0)
    # DPT is not computed here; deferred to ensure_pseudotime
    return adata

def paga_connectivity_binary(adata, key, threshold=0.03):
    sc.tl.paga(adata, groups=key)
    M = adata.uns["paga"]["connectivities"]
    M = M.A if issparse(M) else np.array(M)
    return (M >= threshold).astype(int), M

def cluster_centroids_umap(adata,key):
    um = adata.obsm["X_umap"]
    labs = adata.obs[key].astype(str).values
    cats = sorted(np.unique(labs), key=lambda x: int(x) if x.isdigit() else x)
    cents = [um[labs == c].mean(axis=0) for c in cats]
    return np.vstack(cents), cats

def laplacian_spectral_distance(A1, A2, k=10):
    def topk(A, k):
        D = np.diag(A.sum(1))
        with np.errstate(divide='ignore'):
            D_inv_sqrt = np.diag(1.0/np.sqrt(np.maximum(np.diag(D), 1e-12)))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        w = np.sort(np.real(np.linalg.eigvalsh(L)))
        return w[:min(k, len(w))]
    k_use = min(k, A1.shape[0], A2.shape[0])
    e1, e2 = topk(A1, k_use), topk(A2, k_use)
    m = min(len(e1), len(e2))
    return float(np.linalg.norm(e1[:m] - e2[:m]))

def shortest_path_spearman(G1, G2):
    def spd(G):
        dist = dict(nx.all_pairs_shortest_path_length(G))
        nodes = list(G.nodes())
        M = np.full((len(nodes), len(nodes)), np.inf)
        idx = {n:i for i,n in enumerate(nodes)}
        for u,d in dist.items():
            i = idx[u]
            for v,l in d.items():
                j = idx[v]; M[i,j] = l
        iu = np.triu_indices_from(M, k=1)
        vals = M[iu]; vals = vals[np.isfinite(vals)]
        return vals
    v1, v2 = spd(G1), spd(G2)
    if len(v1)==0 or len(v2)==0: return np.nan
    q = np.linspace(0.01,0.99,99)
    qa, qb = np.quantile(v1,q), np.quantile(v2,q)
    rho, _ = spearmanr(qa, qb)
    return float(rho)

def build_cluster_graph(A_bin, cats):
    G = nx.from_numpy_array(A_bin)
    mapping = {i: cats[i] for i in range(len(cats))}
    G = nx.relabel_nodes(G, mapping)
    return G

def mst_from_paga(A_cont, A_bin, cents, cats):
    G = nx.Graph()
    for i in range(len(cats)):
        G.add_node(cats[i], pos=cents[i])
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            if A_bin[i,j] > 0:
                w = A_cont[i,j]
                G.add_edge(cats[i], cats[j], weight=float(w))
    if G.number_of_edges()==0:
        return G
    T = nx.maximum_spanning_tree(G, weight="weight")
    return T

def _quad_bezier(p0, p1, p2, n=50):
    t = np.linspace(0,1,n)
    B = (1-t)[:,None]**2*p0 + 2*(1-t)[:,None]*t[:,None]*p1 + (t[:,None]**2)*p2
    return B

def _control_point(a, b):
    m = 0.5*(a+b)
    v = b - a
    if np.linalg.norm(v) < 1e-8:
        return m
    n = np.array([-v[1], v[0]]); n = n/np.linalg.norm(n)
    return m + 0.15*np.linalg.norm(v)*n

def draw_trajectory(ax, adata, T, title, point_size=6, alpha_points=0.6, color_by="dpt_pseudotime"):
    if color_by in adata.obs:
        sc.pl.umap(adata, color=[color_by], ax=ax, show=False, title=title, size=point_size, alpha=alpha_points)
    else:
        sc.pl.umap(adata, ax=ax, show=False, title=title, size=point_size, alpha=alpha_points)
    pos = {n: np.array(T.nodes[n]['pos']) for n in T.nodes}
    for u, v in T.edges():
        p0, p2 = pos[u], pos[v]
        p1 = _control_point(p0, p2)
        B = _quad_bezier(p0, p1, p2, n=80)
        ax.plot(B[:,0], B[:,1], linewidth=3, alpha=0.9)

def elpi_principal_graph_on_umap(adata, nodes=30, mu=0.1, lam=0.01, seed=42):
    X = adata.obsm["X_umap"]
    res = elpigraph.computeElasticPrincipalTree(
        X, NumNodes=nodes, Mu=mu, Lambda=lam, Seed=seed
    )
    G = nx.Graph()
    Y = res["NodePositions"]
    E = res["Edges"]
    for i in range(Y.shape[0]):
        G.add_node(i, pos=Y[i])
    for e in E:
        u = int(e[0]-1); v=int(e[1]-1)
        G.add_edge(u, v)
    return G

def draw_elpi(ax, adata, G, title, point_size=6, alpha_points=0.6, color_by="dpt_pseudotime"):
    if color_by in adata.obs:
        sc.pl.umap(adata, color=[color_by], ax=ax, show=False, title=title, size=point_size, alpha=alpha_points)
    else:
        sc.pl.umap(adata, ax=ax, show=False, title=title, size=point_size, alpha=alpha_points)
    for u, v in G.edges():
        p0 = G.nodes[u]['pos']; p2 = G.nodes[v]['pos']
        p1 = _control_point(p0, p2)
        B = _quad_bezier(p0, p1, p2, n=80)
        ax.plot(B[:,0], B[:,1], linewidth=3, alpha=0.9)

def root_dist_spearman(G1, root1, G2, root2):
    def dist_from_root(G, root):
        d = nx.single_source_shortest_path_length(G, root)
        vals = np.array(list(d.values()), dtype=float)
        return vals

    v1 = dist_from_root(G1, root1)
    v2 = dist_from_root(G2, root2)
    if len(v1) == 0 or len(v2) == 0:
        return np.nan

    q = np.linspace(0.01, 0.99, 99)
    qa = np.quantile(v1, q)
    qb = np.quantile(v2, q)
    rho, _ = spearmanr(qa, qb)
    return float(rho)

def degree_spearman(G1, G2):
    import numpy as np
    from scipy.stats import spearmanr

    d1 = np.array([d for _, d in G1.degree()], dtype=float)
    d2 = np.array([d for _, d in G2.degree()], dtype=float)
    if len(d1) == 0 or len(d2) == 0:
        return np.nan

    q = np.linspace(0.01, 0.99, 99)
    qa = np.quantile(d1, q)
    qb = np.quantile(d2, q)
    rho, _ = spearmanr(qa, qb)
    return float(rho)

def metrics_from_graphs(A_bin_r, A_cont_r, cats_r,
                        A_bin_g, A_cont_g, cats_g,
                        adata_r, adata_g, key,
                        pt_key="dpt_pseudotime"):
    Gr = build_cluster_graph(A_bin_r, cats_r)
    Gg = build_cluster_graph(A_bin_g, cats_g)
    # ej = graph_edge_jaccard(A_bin_r, A_bin_g)
    sd = laplacian_spectral_distance(A_cont_r, A_cont_g, k=min(10, A_cont_r.shape[0], A_cont_g.shape[0]))
    sp = shortest_path_spearman(Gr, Gg)
    ds = degree_spearman(Gr, Gg)

    mean_pt = adata_r.obs.groupby(key)[pt_key].mean()
    root1 = mean_pt.idxmin()
    mean_pt = adata_g.obs.groupby(key)[pt_key].mean()
    root2 = mean_pt.idxmin()
    rds = root_dist_spearman(Gr, root1, Gg, root2)

    return {
        "laplacian_spectral_L2": sd,
        "shortest_path_spearman": sp,
        "degree_spearman": ds,
        "root_dist_spearman": rds
    }

# ------------------ MAIN ENTRY (function) ------------------
def analysis_topo(adata_r, adata_g, time_key, outdir="results_trajplot",
                  n_pcs=50, neighbors=30, paga_threshold=0.03, seed=42,
                  use_elpi=True, pt_key="dpt_pseudotime"):
    os.makedirs(outdir, exist_ok=True)

    # 1) Align genes & prepare obs types
    adata_r, adata_g = intersect_genes(adata_r, adata_g)

    # 2) Minimal preprocessing
    adata_r = basic_preprocess(adata_r, n_pcs, neighbors, seed)
    adata_g = basic_preprocess(adata_g, n_pcs, neighbors, seed)

    for ad_ in (adata_r, adata_g):
        if time_key in ad_.obs:
            ad_.obs[time_key] = pd.Categorical(ad_.obs[time_key], ordered=True)

    # 3) Ensure pseudotime exists (avoids KeyError)
    ensure_pseudotime(adata_r, time_key=time_key, pt_key=pt_key)
    ensure_pseudotime(adata_g, time_key=time_key, pt_key=pt_key)

    # 4) PAGA & cluster summaries
    A_rb, A_r = paga_connectivity_binary(adata_r, key=time_key, threshold=paga_threshold)
    A_gb, A_g = paga_connectivity_binary(adata_g, key=time_key, threshold=paga_threshold)
    C_r, cats_r = cluster_centroids_umap(adata_r, key=time_key)
    C_g, cats_g = cluster_centroids_umap(adata_g, key=time_key)

    # 5) Metrics (pass pt_key)
    metrics = metrics_from_graphs(
        A_rb, A_r, cats_r, A_gb, A_g, cats_g, adata_r, adata_g, key=time_key, pt_key=pt_key
    )
    pd.Series(metrics).to_csv(os.path.join(outdir, "metrics_summary.csv"))

    # 6) Principal trajectories
    if use_elpi and _have_elpi:
        Gtraj_r = elpi_principal_graph_on_umap(adata_r, nodes=min(40, max(5, len(cats_r)*3)), seed=seed)
        Gtraj_g = elpi_principal_graph_on_umap(adata_g, nodes=min(40, max(5, len(cats_g)*3)), seed=seed)
        plot_fn = "traj_elpi"
    else:
        T_r = mst_from_paga(A_r, A_rb, C_r, cats_r)
        T_g = mst_from_paga(A_g, A_gb, C_g, cats_g)
        for n in T_r.nodes:
            idx = cats_r.index(n); T_r.nodes[n]['pos'] = C_r[idx]
        for n in T_g.nodes:
            idx = cats_g.index(n); T_g.nodes[n]['pos'] = C_g[idx]
        Gtraj_r, Gtraj_g = T_r, T_g
        plot_fn = "traj_mst"

    # 7) Plots: DPT (or pt_key) & time_key
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if plot_fn == "traj_elpi":
        draw_elpi(axes[0], adata_r, Gtraj_r, f"REAL: trajectory ({pt_key})", color_by=pt_key, point_size=25)
        draw_elpi(axes[1], adata_g, Gtraj_g, f"GEN: trajectory ({pt_key})", color_by=pt_key, point_size=25)
    else:
        draw_trajectory(axes[0], adata_r, Gtraj_r, f"REAL: trajectory ({pt_key})", color_by=pt_key, point_size=25)
        draw_trajectory(axes[1], adata_g, Gtraj_g, f"GEN: trajectory ({pt_key})", color_by=pt_key, point_size=25)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{plot_fn}_pt.pdf")); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if plot_fn == "traj_elpi":
        draw_elpi(axes[0], adata_r, Gtraj_r, f"REAL: trajectory ({time_key})", color_by=time_key, point_size=25)
        draw_elpi(axes[1], adata_g, Gtraj_g, f"GEN: trajectory ({time_key})", color_by=time_key, point_size=25)
    else:
        draw_trajectory(axes[0], adata_r, Gtraj_r, f"REAL: trajectory ({time_key})", color_by=time_key, point_size=25)
        draw_trajectory(axes[1], adata_g, Gtraj_g, f"GEN: trajectory ({time_key})", color_by=time_key, point_size=25)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{plot_fn}_time.pdf")); plt.close()


def match_pattern_with_free_tail(traj, pattern):
    """
    Requirements:
    - The trajectory prefix must match the segmented pattern:
        pattern = [A, B, C]
        valid prefix: A* B* C* (each segment must appear at least once)
    - No cell type outside the pattern is allowed in the prefix region
    - After the pattern is fully matched, the tail can be anything
    - All types in the pattern must appear (each at least once) in the correct order
    """
    if not pattern:
        return False

    order = {p: i for i, p in enumerate(pattern)}
    seen = [False] * len(pattern)
    p_idx = 0  # currently in segment pattern[p_idx]

    for cell in traj:
        if cell not in order:
            # foreign cell type encountered before the pattern is fully matched
            if all(seen):
                # pattern fully matched; tail can be anything
                return True
            else:
                return False

        idx = order[cell]

        # backtracking is forbidden, e.g. for pattern = [HSC, MPP]
        # HSC appearing again after MPP (while still within the pattern region) is invalid
        if idx < p_idx:
            return False

        # attempt to advance to the next segment
        if idx > p_idx:
            # must advance exactly one step; skipping is not allowed (e.g. A -> C)
            if idx == p_idx + 1:
                p_idx = idx
            else:
                return False

        # mark this pattern element as seen
        seen[idx] = True

        # if all pattern elements have appeared at least once,
        # the pattern is fully matched; return True immediately
        if all(seen):
            return True

    # full trajectory exhausted without matching all pattern elements → no match
    return False

def replace_query_with_nn_from_ref(
    adata_ref,
    adata_query,
    pca_key: str = "X_pca_ref",
    copy_obs_query: bool = True,
):
    """
    Find the nearest neighbour in ref for each query cell in PCA space,
    then construct a “surrogate” AnnData from those ref cells (same cell count as query).

    Parameters
    ----
    adata_ref : AnnData
        Reference data; must have adata_ref.obsm[pca_key].
    adata_query : AnnData
        Query data to be “replaced”; must have adata_query.obsm[pca_key].
    pca_key : str
        Key storing PCA coordinates, e.g. "X_pca_ref" or "X_pca_joint".
    copy_obs_query : bool
        If True, retain some query information in the new adata.obs
        (extra columns added for traceability).

    Returns
    ----
    adata_new : AnnData
        New AnnData with X / var from ref, aligned one-to-one with query.
    nn_indices : np.ndarray
        Index of the nearest ref neighbour for each query cell (row index in adata_ref).
    """

    # 1) get PCA coordinates
    X_ref = adata_ref.obsm[pca_key]
    X_q   = adata_query.obsm[pca_key]

    # 2) build nearest-neighbour index on ref
    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nbrs.fit(X_ref)

    # 3) find nearest ref index for each query cell
    dist, idx = nbrs.kneighbors(X_q)  # idx shape: (n_query, 1)
    nn_indices = idx[:, 0]            # flatten to (n_query,)

    # 4) build new matrix / obs / obsm from nearest ref cells
    X_new = adata_ref.X[nn_indices, :]

    # obs: start from ref obs
    obs_new = adata_ref.obs.iloc[nn_indices].copy()
    obs_new.index = adata_query.obs_names  # replace index with query cell names

    # optional: merge query info for comparison
    if copy_obs_query:
        for col in adata_query.obs.columns:
            obs_new[f"query_{col}"] = adata_query.obs[col].values

    # var: use ref var directly (assumes ref/query gene sets are already aligned)
    var_new = adata_ref.var.copy()

    # build new AnnData
    adata_new = sc.AnnData(X=X_new, obs=obs_new, var=var_new)

    # 5) optional: copy embeddings from ref (PCA / UMAP, etc.)
    for key in adata_ref.obsm_keys():
        if adata_ref.obsm[key].shape[0] == adata_ref.n_obs:
            adata_new.obsm[key] = adata_ref.obsm[key][nn_indices, :].copy()

    # store nearest-neighbour indices / distances for debugging
    adata_new.obsm["nn_index_in_ref"] = nn_indices[:, None]
    adata_new.obsm["nn_dist_in_pca"] = dist

    return adata_new, nn_indices