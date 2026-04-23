import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as ADataset
from datasets import load_from_disk, concatenate_datasets, DatasetDict

import torch
import os, glob
import pandas as pd
from loguru import logger
from tqdm import tqdm

def load_and_concatenate_shards(parent_dir: str, expect_features=None):
    """读取分片并合并成一个 Dataset（零拷贝合并）"""
    dirs = [d for d in glob.glob(os.path.join(parent_dir, "part_*")) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"No shards under {parent_dir}")
    # 按编号排序
    import re as _re
    dirs = sorted(dirs, key=lambda p: int(_re.search(r"part_(\d+)", p).group(1)))
    # parts = [load_from_disk(p) for p in tqdm(dirs, desc="Loading datasets")]
    parts = []
    for p in tqdm(dirs, desc="Loading datasets"):
        try:
            parts.append(load_from_disk(p))
        except:
            print(f"failed to load the shard: {p}")
        # snap(f"after shard {p}")

    return concatenate_datasets(parts)


def load_data(
    *,
    data_dir,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = load_and_concatenate_shards(data_dir)  
    dataset = mixDataTypeTargetDataset_scbasecount(data_folders=[data_dir], dataset=dataset, mode='train') 

    return dataset

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


class mixDataTypeTargetDataset_scbasecount(Dataset): # DatasetList里可以给多个huggingface dataset
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
                 bin_type: str = 'cell_cell_dep', # cell_cell_dep for related bin; cell_ind for individual bin
                 data_types: list = ['velocity','perturb'], # sc-rna, velocity, perturb
                 mix_ratio: list = [1.0],
                 topGene_table_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan/debug/topGenes_three_sections/',
                 topDEGs_num: list = [20],
                 topDEGs_ratio: list = [1.0],
                 infer_mode: bool = False,
                 gene_cache_path: str = 'gene_cache.json',
                 dataset: ADataset = None,
                ):
        
        self.datasets = []
        # 生成dataset 所有数据集处理
        with open("/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan_AR_traj/build_dataset/train_list.txt", "r", encoding="utf-8") as f:
            train_samples = [line.strip() for line in f]

        with open("/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan_AR_traj/build_dataset/test_list.txt", "r", encoding="utf-8") as f:
            test_samples = [line.strip() for line in f]

        chosen_data = [sample.split('.')[0] for sample in test_samples]
        if mode == 'train':
            self.datasets.append(filter_by_names(dataset, dataset_names=chosen_data, train_flag=True,  num_proc=4))
        else:
            self.datasets.append(filter_by_names(dataset, dataset_names=chosen_data, train_flag=False,  num_proc=4))

        self.mix_ratio = mix_ratio
        # 初始化每个数据集的长度，方便索引
        self.dataset_lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.dataset_lengths)
        self.reference_gene = pd.read_csv('/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan/trainer/OS_scRNA_gene_index.18791.tsv',sep='\t')['gene_name'].values #19264

#         self.device = 'cuda'
        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.success('Loading data from {} succeed'.format(data_folders))
            logger.info(f'vocab size is {self.reference_gene.shape[0]}')
            logger.info(f'dataset length is {self.total_length}')
        
            
    def __len__(self):            
        weighted_lengths = [ratio * length for ratio, length in zip(self.mix_ratio, self.dataset_lengths)]
        return int(sum(weighted_lengths))
        
    def __getitem__(self, idx_num):

        dataset_idx = random.choices(
            population=range(len(self.datasets)),
            weights=self.mix_ratio,
            k=1
        )[0]

        idx_num = idx_num % self.dataset_lengths[dataset_idx]
        dataset = self.datasets[dataset_idx]
        cell_data = dataset[idx_num]
        gene_names = cell_data['expressed_genes']
        expr_values = cell_data['expressed_values']

        expr_vec = pd.Series(expr_values, index=gene_names).reindex(self.reference_gene, fill_value=0).to_numpy()
        out_dict = {}
        out_dict['exp'] = torch.tensor(expr_vec)

        return out_dict