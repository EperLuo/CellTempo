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
    将 adata 的基因映射到参考基因列表 ref_genes 上。
    - 如果基因缺失，则补零列。
    - 如果有多余基因，则会被去除。
    - 输出的 adata.var 会按 ref_genes 顺序排列。

    参数:
        adata: AnnData 对象
        ref_genes: list[str]，参考基因列表（目标基因顺序）
    返回:
        一个新的 AnnData 对象
    """
    # 现有基因名
    current_genes = np.array(adata.var_names)

    # 找出交集和索引映射
    intersect_genes = np.intersect1d(current_genes, ref_genes)
    missing_genes = [g for g in ref_genes if g not in current_genes]

    print(f"✅ {len(intersect_genes)} genes matched, "
          f"{len(missing_genes)} missing from adata.")

    # 取出交集基因的表达矩阵
    adata_aligned = adata[:, intersect_genes].copy()

    # 如果存在缺失基因，用0填充这些列
    if missing_genes:
        import scipy.sparse as sp
        n_cells = adata_aligned.n_obs
        zero_mat = sp.csr_matrix((n_cells, len(missing_genes)))
        from anndata import AnnData
        adata_missing = ad.AnnData(X=zero_mat)
        adata_missing.var_names = missing_genes
        adata_missing.obs_names = adata_aligned.obs_names
        adata_aligned = ad.concat([adata_aligned, adata_missing], axis=1)

    # 按 ref_genes 顺序重新排列
    adata_aligned = adata_aligned[:, ref_genes].copy()
    adata_aligned.obs = adata.obs

    return adata_aligned


def collate_fn(batch):
    # 分别提取tokens和values
    tokens = [torch.tensor(item['tokens']) for item in batch]
    values = [torch.tensor(item['values']) for item in batch]
    data_len = torch.tensor([torch.tensor(item['trunc_full_len']) for item in batch])
    c1_len = torch.tensor([torch.tensor(item['c1_len']) for item in batch])
    c2_start = torch.tensor([torch.tensor(item['c2_start']) for item in batch])
    
    
    # 生成标签，这里简单地将序列向左移动一位
    y_t = [torch.cat([t[1:], torch.tensor([0])]) for t in tokens]  # 对tokens生成下一个词的标签
    y_v = [torch.cat([v[1:], torch.tensor([0])]) for v in values]  # 对values做同样处理

    # 填充或截断
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    values_padded = pad_sequence(values, batch_first=True, padding_value=0)
    y_t_padded = pad_sequence(y_t, batch_first=True, padding_value=0)
    y_v_padded = pad_sequence(y_v, batch_first=True, padding_value=0)
    
    # 根据c1_len创建一个形状与tokens_padded相同的张量，元素由0和1组成
    cell_pos = torch.zeros_like(tokens_padded)
    for i, c1_ln in enumerate(c1_len):
        c2_st = c2_start[i]
        cell_pos[i, c1_ln:c2_st] = 1
        cell_pos[i, c2_st:] = 2

    # 返回处理后的批次数据
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
    
    # 生成标签，这里简单地将序列向左移动一位
    y_t = [torch.cat([torch.tensor(t[1:]), torch.tensor([0])]) for t in tokens]

    # 填充或截断
    end_token = tokens[0][-1]
    tokens_padded = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=end_token)
    y_t_padded = pad_sequence(y_t, batch_first=True, padding_value=end_token)

    # padding值设为0
    cell_pos = pad_sequence([torch.tensor(c) for c in cell_pos_list], batch_first=True, padding_value=0)

    # 返回处理后的批次数据
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
    由于每个批次内的样本长度相同，因此无需 padding。
    我们可以直接堆叠各个字段。
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
    
    # 确保所有切片后的长度一致
    slice_lengths = [item['c2_start'] for item in batch]
    assert len(set(slice_lengths)) == 1, "Slice lengths are not consistent."
    
    # c2_start是一个cell的长度，+2是cell_id和<S>
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
    
    # 使用和train_target相同的逻辑处理cell_pos
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)  # 无 padding，全部为 True

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
    """读取分片并合并成一个 Dataset（零拷贝合并）"""
    dirs = [d for d in glob.glob(os.path.join(parent_dir, "part_*")) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"No shards under {parent_dir}")
    # 按编号排序
    import re as _re
    dirs = sorted(dirs, key=lambda p: int(_re.search(r"part_(\d+)", p).group(1)))

    parts = [load_from_disk(p) for p in tqdm(dirs, desc="Loading datasets")]

    return concatenate_datasets(parts)

def filter_by_names(dataset, dataset_names, train_flag=True, num_proc=8, batch_size=100_000):
    """
    dataset: Dataset 或 DatasetDict（若是 DatasetDict，则对每个 split 过滤）
    dataset_names: 要“保留为测试集”的数据集名列表
    train_flag: True 取训练集（not in 名单），False 取测试集（in 名单）
    num_proc: 并行进程数
    batch_size: batched 过滤时的批大小（可按内存调）
    """
    names = set(dataset_names)

    def keep_batch(batch):
        # 向量化判断，返回布尔列表
        if train_flag:
            return [nm.split('/')[-1] not in names for nm in batch]
        else:
            return [nm.split('/')[-1] in names for nm in batch]

    if isinstance(dataset, DatasetDict):
        # 对每个 split 分别过滤
        return DatasetDict({
            split: ds.filter(
                keep_batch, input_columns=["dataset_name"],
                batched=True, batch_size=batch_size,
                num_proc=num_proc, desc=f"Filter {split} ({'train' if train_flag else 'test'})"
            )
            for split, ds in dataset.items()
        })
    else:
        # 单个 Dataset
        return dataset.filter(
            keep_batch, input_columns=["dataset_name"],
            batched=True, batch_size=batch_size,
            num_proc=num_proc, desc=f"Filter ({'train' if train_flag else 'test'})"
        )

class scBasetraj_vq(Dataset): # DatasetList里可以给多个huggingface dataset
    ## 目前兼容： Velocity ✅，scperturb✅
    ## 对于第二个细胞，我不再使用全长基因，而是只把Top 100的 genes 放进去。
    ## 解决从哪儿获取Top 100 list的问题。✅
    ## FIXME 这样其实只兼容perturbation任务，不兼容velocity了，但也可以先都做完预训练，然后再做这个。
    ## FIXME 给中间连接token加上positional encoding，这样就可以区分前后了。不然第二个cell不知道要按什么顺序生成。

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # 只有velocity需要这个，来globally找next cell
                 mode: str = 'train',
                 global_dataset: str = 'velo_dataset_all', # 只有velocity需要这个，来globally找next cell
                 data_types: list = ['trajectory','perturb'], # sc-rna, velocity, perturb
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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前进程的 GPU 编号
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
        while traj[0] not in self.all_cell_name:  # 有一些数据在预处理的时候被丢掉了
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

        # 保存instruction信息
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
            'c1_len': c1_len, # 不带 inter tokens
            'c2_start': c1_len, # 加上inter_tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # 新增：保存instruction信息
        }
        return concated_cell

class h5ad_data_vq(Dataset): # DatasetList里可以给多个huggingface dataset
    ## 目前兼容： Velocity ✅，scperturb✅
    ## 对于第二个细胞，我不再使用全长基因，而是只把Top 100的 genes 放进去。
    ## 解决从哪儿获取Top 100 list的问题。✅
    ## FIXME 这样其实只兼容perturbation任务，不兼容velocity了，但也可以先都做完预训练，然后再做这个。
    ## FIXME 给中间连接token加上positional encoding，这样就可以区分前后了。不然第二个cell不知道要按什么顺序生成。

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # 只有velocity需要这个，来globally找next cell
                 mode: str = 'train',
                 data_types: list = ['trajectory','perturb'], # sc-rna, velocity, perturb
                 global_dataset: str = 'velo_dataset_all', # 只有velocity需要这个，来globally找next cell
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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前进程的 GPU 编号
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

        # 保存instruction信息
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
            'c1_len': c1_len, # 不带 inter tokens
            'c2_start': c1_len, # 加上inter_tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # 新增：保存instruction信息
        }
        return concated_cell


class Tahoe100m_vq(Dataset): # DatasetList里可以给多个huggingface dataset
    ## 目前兼容： Velocity ✅，scperturb✅
    ## 对于第二个细胞，我不再使用全长基因，而是只把Top 100的 genes 放进去。
    ## 解决从哪儿获取Top 100 list的问题。✅
    ## FIXME 这样其实只兼容perturbation任务，不兼容velocity了，但也可以先都做完预训练，然后再做这个。
    ## FIXME 给中间连接token加上positional encoding，这样就可以区分前后了。不然第二个cell不知道要按什么顺序生成。

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 n_express_level: int = 10,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # 只有velocity需要这个，来globally找next cell
                 mode: str = 'train',
                 global_dataset: str = 'velo_dataset_all', # 只有velocity需要这个，来globally找next cell
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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前进程的 GPU 编号
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

        # 保存instruction信息
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
            'c1_len': c1_len, # 不带 inter tokens
            'c2_start': c1_len, # 加上inter_tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # 新增：保存instruction信息
        }
        return concated_cell


class h5ad_traj_vq(Dataset): # DatasetList里可以给多个huggingface dataset

    def __init__(self,
                 data_folders: list = ['path1','path2'],
                 dataset_names: list = ['name1','name2'],
                 crop_train_length: int = 6000,
                 meta_info_name: str = 'mix_meta_info.json',
                 mapping_dict: str = 'velo_mapping_dict.json', # 只有velocity需要这个，来globally找next cell
                 mode: str = 'train',
                 global_dataset: str = 'velo_dataset_all', # 只有velocity需要这个，来globally找next cell
                 data_types: list = ['trajectory','perturb'], # sc-rna, velocity, perturb
                 dataset: ADataset = None,
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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前进程的 GPU 编号
        device = torch.device(f"cuda:{local_rank}")
        print('current device: ', device)
        self.vq_model = VQModel.from_pretrained(vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')#.to(device=device)

        self.global_dataset = map_adata_to_reference_genes(self.global_dataset, self.reference_gene)
        HSC_to_CLP_up = [
            "SPI1",    # PU.1：驱动淋系启动（但强度中等，过高会抑制 B/T）
            "IKZF1",   # Ikaros：CLP 定义基因
            "IKZF3",   # Aiolos：淋系强化
            "TCF3",    # E2A：B/T 细胞基本 TF
            "EBF1",    # B lineage 指定基因
            "FLT3",    # LMPP/CLP marker，促进早期淋系扩增
            "IL7R",    # IL7R+ 是 CLP 标志
            "DNTT",    # TdT，强淋系标志
            "LYL1",    # LMPP/CLP priming
            "LMO2",
        ]
        HSC_to_CLP_down = [
            "GATA1",   # 抑制红系/巨核方向
            "GATA2",   # 推髓系 priming
            "CEBPA",   # 髓系 master regulator
            "CEBPB",
            "GFI1",    # granulocyte program
            "CSF1R",   # macrophage receptor
        ]
        HSC_to_DC_up = [
            "IRF8",      # DC1 关键因子；高 IRF8 → DC1
            "BATF3",     # cDC1 master regulator
            "ID2",       # 抑制 pDC 程序，推动 DC1
            "SPI1",      # PU.1：DC通用前向调控
            "FLT3",      # FLT3L 信号促进 DC 生成
            "ZBTB46",    # classical DC signature gene
        ]
        HSC_to_DC_down = [
            "GATA1",
            "GATA2",
            "CEBPA",     # 高 CEBPA 会阻止 DC 分化转为 granulocyte
            "TCF3",      # T/B cell program，需抑制
            "EBF1",      # B 系统 TF
        ]
        HSC_to_Mono_up = [
            "IRF8",      # monocyte / DC2 program
            "CEBPB",     # monocyte & macrophage differentiation
            "SPI1",      # PU.1：高水平偏向 monocyte/macrophage
            "KLF4",      # 强制 KLF4 上调 → 单核命运
            "CSF1R",     # M-CSF receptor，推动 monocyte fate
            "RUNX1",
            "LYZ",       # lysozyme, monocyte marker
        ]
        HSC_to_Mono_down = [
            "IKZF1", "IKZF3",  # 淋系程序
            "TCF3", "EBF1",    # B cell TF
            "GATA1", "GATA2",  # 红系/巨核
        ]

        HSC_to_Ery_up = [
            "GATA1",
            "KLF1",
            "GFI1B",
            "NFE2",
            "ZFPM1", # FOG1
        ]
        HSC_to_Ery_down = [
            "SPI1", "IKZF1", "IKZF3", "TCF3", "EBF1", "FLT3", "IL7R"
        ]


        self.global_dataset.X = self.global_dataset.X.toarray() if sp.issparse(self.global_dataset.X) else self.global_dataset.X
        self.global_dataset[self.global_dataset.obs['clusters']=='HSC_1'].X = self.amplify_genes(self.global_dataset[self.global_dataset.obs['clusters']=='HSC_1'],HSC_to_CLP_up,add=2,mul=8)
        self.global_dataset[self.global_dataset.obs['clusters']=='HSC_1'].X = self.amplify_genes(self.global_dataset[self.global_dataset.obs['clusters']=='HSC_1'],HSC_to_CLP_down,add=1,mul=0)
        self.global_dataset[self.global_dataset.obs['clusters']=='HSC_2'].X = self.amplify_genes(self.global_dataset[self.global_dataset.obs['clusters']=='HSC_2'],HSC_to_CLP_up,add=2,mul=8)
        self.global_dataset[self.global_dataset.obs['clusters']=='HSC_2'].X = self.amplify_genes(self.global_dataset[self.global_dataset.obs['clusters']=='HSC_2'],HSC_to_CLP_down,add=1,mul=0)
        self.global_dataset = self.global_dataset.X

        with open("/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan_AR_traj/trainer/generation/data/bone_marrow.pkl", "rb") as f:
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

        # 保存instruction信息
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
            'c1_len': c1_len, # 不带 inter tokens
            'c2_start': c1_len, # 加上inter_tokens
            'trunc_full_len': full_length,
            'cell_pos': cell_pos,
            'instructions': instruction,  # 新增：保存instruction信息
        }
        return concated_cell

    def amplify_genes(self, adata, gene_list, add=1.0, mul=5.0):
        """对指定基因进行表达提升：先 +add，再 ×mul"""
        # 基因名 → index
        gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

        # 找到基因索引（排除不存在的基因）
        idx = [gene_to_idx[g] for g in gene_list if g in gene_to_idx]
        if not idx:
            print("⚠️ 没有基因匹配！")
            return adata

        X = adata.X

        # 稀疏矩阵处理
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