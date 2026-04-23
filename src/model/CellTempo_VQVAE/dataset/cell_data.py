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
    """Load shards and concatenate into a single Dataset (zero-copy merge)."""
    dirs = [d for d in glob.glob(os.path.join(parent_dir, "part_*")) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"No shards under {parent_dir}")
    # sort by shard index
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


class mixDataTypeTargetDataset_scbasecount(Dataset): # Accepts multiple HuggingFace datasets
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
                 bin_type: str = 'cell_cell_dep', # cell_cell_dep for related bin; cell_ind for individual bin
                 data_types: list = ['velocity','perturb'], # supported types: sc-rna, velocity, perturb
                 mix_ratio: list = [1.0],
                 topGene_table_path: str = '/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan/debug/topGenes_three_sections/',
                 topDEGs_num: list = [20],
                 topDEGs_ratio: list = [1.0],
                 infer_mode: bool = False,
                 gene_cache_path: str = 'gene_cache.json',
                 dataset: ADataset = None,
                ):
        
        self.datasets = []
        # generate dataset — process all datasets
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
        # initialise dataset lengths for indexing
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