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
    Since all samples within a batch have the same length, no padding is needed.
    Fields can be stacked directly.
    """
    if 'target_id' in batch[0].keys():
        prefix_num = batch[0]['target_id']
    else:
        prefix_num = 1
    
    if not batch:
        return None
    c2_start_values = [item['c2_start'] for item in batch]
    unique_c2_start = set(c2_start_values)
    if len(unique_c2_start) != 1:
        print(f"Different c2_start values in batch: {unique_c2_start}")
    assert len(unique_c2_start) == 1, f"Batch samples have different c2_start values: {unique_c2_start}"
    
    # ensure all sliced lengths are consistent
    slice_lengths = [item['c2_start'] for item in batch]
    assert len(set(slice_lengths)) == 1, "Slice lengths are not consistent."
    
    # c2_start is one cell's length; +2 for the cell_id token and <S> token
    input_ids = torch.stack([torch.tensor(item['tokens'][:item['c2_start']*prefix_num+2]) for item in batch], dim=0) 
    # x_expr = torch.stack([torch.tensor(item['values'][:item['c2_start']]) for item in batch], dim=0)
    c1_len = torch.tensor([item['c1_len'] for item in batch], dtype=torch.long)
    c2_start = torch.tensor([item['c2_start'] for item in batch], dtype=torch.long)

    cell_pos = torch.stack([torch.tensor(item['cell_pos'][:item['c2_start']*prefix_num+2]) for item in batch], dim=0)

    idx = [item['idx'] for item in batch]
    
    token_labels = pad_sequence([
        torch.tensor(item['tokens'][item['c2_start']*prefix_num+2:]) for item in batch
    ], batch_first=True, padding_value=-100)
    
    # expr_labels = pad_sequence([
    #     torch.tensor(item['values'][item['c2_start']:]) for item in batch
    # ], batch_first=True, padding_value=-100)
    
    # build attention_mask using the same logic as train_target
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)  # no padding — all True

    return {
        'input_ids': input_ids,
        # 'x_expr': x_expr,
        'c1_len': c1_len,
        'c2_start': c2_start,
        'cell_pos': cell_pos,
        'attention_mask': attention_mask,
        'token_labels': token_labels,
        # 'expr_labels': expr_labels,
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
        with open(os.path.join(data_folders[velo_data_indx], mapping_dict), 'r') as f:
            self.cell_name_to_num = json.load(f)
            self.all_cell_name = self.cell_name_to_num.keys()

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
                 vq_vae_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel'
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
        
        self.reference_gene = pd.read_csv('OS_scRNA_gene_index.18791.tsv',sep='\t')['gene_name'].values
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index of the current process
        device = torch.device(f"cuda:{local_rank}")
        print('current device: ', device)
        self.vq_model = VQModel.from_pretrained(vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')#.to(device=device)

        pairs_path = os.path.join(data_folders[0],f'pairs_{mode}.json.gz')
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

        gene_metadata = load_dataset("vevotx/Tahoe-100M", name="gene_metadata", split="train")
        self.gene_vocab = {}
        for entry in gene_metadata:
            self.gene_vocab[entry["token_id"]]= entry["gene_symbol"]
        sorted_vocab_items = sorted(self.gene_vocab.items())
        token_ids, gene_names = zip(*sorted_vocab_items)
        self.token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

            
    def __len__(self):            
        return len(self.pairs)
        
    def __getitem__(self, idx_num):

        pair = self.pairs[self.perturb_key[idx_num]]
        pert_id = random.choice(pair['pert_ids'])
        ctrl_id = random.choice(pair['ctrl_ids'])
        traj = [ctrl_id, pert_id]
        
        processed_cell = self.extract_gene_and_expr(traj,)
        paired_cell = self.concat_and_trunc_cell(processed_cell, trunc = self.crop_train_length) # if autoregressive
        paired_cell['idx'] = idx_num

        return paired_cell
    
    def get_next_cell_velo(self, idx_num, global_dataset):

        cell_data = global_dataset[idx_num]
        gene_names = cell_data['genes']
        expr_values = cell_data['expressions']
        drug=cell_data['drug']
        smiles=cell_data['canonical_smiles']
        plate=cell_data['plate']
        cell_line_id=cell_data['cell_line_id']

        if expr_values[0] < 0: 
            gene_names = gene_names[1:]
            expr_values = expr_values[1:]
        
        gene_names = [self.gene_vocab[gene] for gene in gene_names]

        return gene_names, expr_values, drug, smiles, plate, cell_line_id
  
    def extract_gene_and_expr(self, idx_num):
        expr_values = [] 

        for next_cell_name in idx_num:
            gene_names_next_cell, expr_values_next_cell, drug, smiles, plate, cell_line_id = self.get_next_cell_velo(next_cell_name, self.global_dataset)
            # expr_values_next_cell = [expr for gene, expr in zip(gene_names_next_cell, expr_values_next_cell) if gene in self.token_id_to_col_idx]
            expr_values_next_cell = torch.tensor(pd.Series(expr_values_next_cell, index=gene_names_next_cell).reindex(self.reference_gene, fill_value=0).values, dtype=torch.float32)
            # gene_names.append(gene_names_next_cell)
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
        }

        return processed_cell
    
    def concat_and_trunc_cell(self, processed_cell, trunc = True):
        
        start_tokens = ['<S>']

        # store instruction info
        instruction = processed_cell['smile']
        end_tokens = ['<E>']
        token_cell = processed_cell['values'].astype(str) #[str(index) for index in processed_cell['values']]


        tokens = [processed_cell['plate'].astype(str),processed_cell['cell_line'].astype(str),'drug']
        cell_pos = [1,1,1]
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
        with open(os.path.join(data_folders[velo_data_indx], mapping_dict), 'r') as f:
            self.cell_name_to_num = json.load(f)
            self.all_cell_name = self.cell_name_to_num.keys()

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
                mask = self.global_dataset.obs['clusters'] == cluster
                self.global_dataset[mask].X = self.amplify_genes(
                    self.global_dataset[mask], gene_list, add=add, mul=mul
                )
            self.global_dataset = self.global_dataset.X

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
    
    def concat_and_trunc_cell(self, processed_cell, data_type, trunc = True):
        
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
            return adata

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