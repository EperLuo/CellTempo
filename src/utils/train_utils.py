import os, glob
from tqdm import tqdm

# import model and data modules
# from utils.dataset import mixDataTypeTargetDataset
from .dataset import Tahoe100m_vq, scBasetraj_vq, h5ad_data_vq, h5ad_traj_vq
from datasets import load_dataset,load_from_disk, concatenate_datasets

import psutil, gc
p = psutil.Process(os.getpid())

def snap(tag):
    mi = p.memory_info()
    try:
        fds = p.num_fds()
    except Exception:
        fds = -1
    print(f"[{tag}] RSS={mi.rss/1024/1024:.1f}MB, FDs={fds}")

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

def get_dataset_config_from_yaml(yaml_config):
    """Extract dataset-related parameters from a YAML config dict."""
    return {
        "data_folders": yaml_config["data_folders"],
        "dataset_names": yaml_config["dataset_names"],
        "meta_info_name": yaml_config["meta_info_name"],
        "block_size": yaml_config["block_size"],
        "mapping_dict": yaml_config["mapping_dict"],
        "global_dataset": yaml_config["global_dataset"],
        "data_types": yaml_config["data_types"],
        "vq_vae_path": yaml_config.get("vq_vae_path", "/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel"),
    }


def initialize_datasets_from_config(dataset_config, skip_train=False):
    """Initialize scBaseTraj training and validation datasets."""
    eval_datasets = {}
    
    dataset = load_and_concatenate_shards(os.path.join(dataset_config["data_folders"][0], dataset_config["dataset_names"][0]))
    
    for ds_idx, ds_name in enumerate(dataset_config["dataset_names"]):
        eval_datasets[ds_name] = {
            'train': scBasetraj_vq(
                data_folders=[dataset_config["data_folders"][ds_idx]],
                dataset_names=[ds_name],
                crop_train_length=dataset_config["block_size"],
                meta_info_name=dataset_config["meta_info_name"],
                mapping_dict=dataset_config["mapping_dict"],
                mode='train',
                data_types=[dataset_config["data_types"][ds_idx]],
                global_dataset=dataset_config["global_dataset"],
                dataset=dataset,
                vq_vae_path=dataset_config["vq_vae_path"]
            ),
            'val': scBasetraj_vq(
                data_folders=[dataset_config["data_folders"][ds_idx]],
                dataset_names=[ds_name],
                crop_train_length=dataset_config["block_size"],
                meta_info_name=dataset_config["meta_info_name"],
                mapping_dict=dataset_config["mapping_dict"],
                mode='test',
                data_types=[dataset_config["data_types"][ds_idx]],
                global_dataset=dataset_config["global_dataset"],
                dataset=dataset,
                vq_vae_path=dataset_config["vq_vae_path"]
            )
        }
    
    # if skip_train is True, return only the val split of eval_datasets
    if skip_train:
        # extract the val split for each dataset
        val_only_datasets = {ds_name: ds_dict['val'] for ds_name, ds_dict in eval_datasets.items()}
        return None, val_only_datasets

    train_dataset = eval_datasets[dataset_config["dataset_names"][0]]['train']

    return train_dataset, eval_datasets

def initialize_datasets_from_config_h5ad(dataset_config, skip_train=False):
    """Initialize h5ad-format validation datasets."""
    eval_datasets = {}

    for ds_idx, ds_name in enumerate(dataset_config["dataset_names"]):
        eval_datasets[ds_name] = {
            'val': h5ad_data_vq(
                data_folders=[dataset_config["data_folders"][ds_idx]],
                dataset_names=[ds_name],
                crop_train_length=dataset_config["block_size"],
                meta_info_name=dataset_config["meta_info_name"],
                mapping_dict=dataset_config["mapping_dict"],
                data_types=[dataset_config["data_types"][ds_idx]],
                mode='test',
                global_dataset=dataset_config["global_dataset"],
                vq_vae_path=dataset_config["vq_vae_path"]
            )
        }
    
    # if skip_train is True, return only the val split of eval_datasets
    if skip_train:
        # extract the val split for each dataset
        val_only_datasets = {ds_name: ds_dict['val'] for ds_name, ds_dict in eval_datasets.items()}
        return None, val_only_datasets

    train_dataset = eval_datasets[dataset_config["dataset_names"][0]]['train']

    return train_dataset, eval_datasets

def initialize_datasets_from_config_h5ad_traj(dataset_config, skip_train=False):
    """Initialize training and validation datasets from h5ad trajectory files."""
    eval_datasets = {}

    for ds_idx, ds_name in enumerate(dataset_config["dataset_names"]):
        eval_datasets[ds_name] = {
            'val': h5ad_traj_vq(
                data_folders=[dataset_config["data_folders"][ds_idx]],
                dataset_names=[ds_name],
                crop_train_length=dataset_config["block_size"],
                meta_info_name=dataset_config["meta_info_name"],
                mapping_dict=dataset_config["mapping_dict"],
                mode='test',
                global_dataset=dataset_config["global_dataset"],
                data_types=[dataset_config["data_types"][ds_idx]],
                vq_vae_path=dataset_config["vq_vae_path"],
                perturb_config=dataset_config.get("perturb_config", None),
                trajectory_pkl=dataset_config.get("trajectory_pkl", None),
            )
        }
    
    # if skip_train is True, return only the val split of eval_datasets
    if skip_train:
        # extract the val split for each dataset
        val_only_datasets = {ds_name: ds_dict['val'] for ds_name, ds_dict in eval_datasets.items()}
        return None, val_only_datasets

    train_dataset = eval_datasets[dataset_config["dataset_names"][0]]['train']

    return train_dataset, eval_datasets

def initialize_datasets_from_config_perturb(dataset_config, skip_train=False):
    """Initialize training and validation datasets for perturbation tasks."""
    eval_datasets = {}
    dataset = load_dataset("parquet", 
                       data_files=os.path.join(dataset_config["data_folders"][0],"data/data/train-*.parquet"), 
                       split="train",
                       cache_dir=os.path.join(dataset_config["data_folders"][0],"hf_cache"))
        
    for ds_idx, ds_name in enumerate(dataset_config["dataset_names"]):
        eval_datasets[ds_name] = {
            'train': Tahoe100m_vq(
                data_folders=[dataset_config["data_folders"][ds_idx]],
                dataset_names=[ds_name],
                crop_train_length=dataset_config["block_size"],
                n_express_level=dataset_config["n_expression_level"],
                meta_info_name=dataset_config["meta_info_name"],
                mapping_dict=dataset_config["mapping_dict"],
                mode='train',
                global_dataset=dataset_config["global_dataset"],
                dataset=dataset,
                vq_vae_path=dataset_config["vq_vae_path"]
            ),
            'val': Tahoe100m_vq(
                data_folders=[dataset_config["data_folders"][ds_idx]],
                dataset_names=[ds_name],
                crop_train_length=dataset_config["block_size"],
                n_express_level=dataset_config["n_expression_level"],
                meta_info_name=dataset_config["meta_info_name"],
                mapping_dict=dataset_config["mapping_dict"],
                mode='testA',
                global_dataset=dataset_config["global_dataset"],
                dataset=dataset,
                vq_vae_path=dataset_config["vq_vae_path"]
            )
        }
    
    # if skip_train is True, return only the val split of eval_datasets
    if skip_train:
        # extract the val split for each dataset
        val_only_datasets = {ds_name: ds_dict['val'] for ds_name, ds_dict in eval_datasets.items()}
        return None, val_only_datasets

    train_dataset = eval_datasets[dataset_config["dataset_names"][0]]['train']

    return train_dataset, eval_datasets