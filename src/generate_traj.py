import wandb

from tqdm import tqdm
import json
from collections import defaultdict

import os
import sys
import yaml
import random
import numpy as np
import torch
from torch.utils.data import Subset
from transformers.trainer_utils import is_main_process
from transformers import  TrainerCallback
import argparse

from torch.utils.data import DataLoader
import torch
from inference import parallel_infer
from utils.train_utils import initialize_datasets_from_config, initialize_datasets_from_config_h5ad, \
    initialize_datasets_from_config_perturb, initialize_datasets_from_config_h5ad_traj
from utils.dataset import collate_fn_infer_traj_vq
from model.CellTempo_backbone import CellTempoConfig, CellTempo_backbone
import multiprocessing


# --------------------- Config loading & processing ---------------------

# append data to a JSON file
def append_to_json_file(data, json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(data)

    with open(json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

def load_yaml_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_model_config_from_yaml(yaml_config, vocab_size):
    """Extract model-related parameters from a YAML config dict."""
    return CellTempoConfig(
        vocab_size=vocab_size,
        block_size = yaml_config["block_size"],
        n_embd=yaml_config["n_embd"],
        n_layer=yaml_config["n_layer"],
        n_head=yaml_config["n_head"],
        dropout=yaml_config["dropout"],
        bias=yaml_config["bias"],
        train_mode=yaml_config["train_mode"],
        cell_pos_num=yaml_config["cell_pos_num"],
        vq_vae_path=yaml_config["vq_vae_path"],
        data_folders=yaml_config["data_folders"], 
        meta_info_name=yaml_config["meta_info_name"],
        use_flash=yaml_config["use_flash"]
    )

def subset_dataset_by_prefix(dataset_val, prefix):
    target_list = []
    j = 0
    print(f'subsetting dataset from dataset val with prefix dataset {prefix}')
    for eval_d in tqdm(dataset_val):
        if eval_d['idx'].startswith(prefix) and eval_d['c2_start'] <= 3000:
            # print(j)
            target_list.append(j)
        j+=1
        if j == len(dataset_val):
            break
        if j == 1000:
            break
    return  target_list


# wrap single-dataset processing into a function
def process_dataset(dataset_name,args):
    print(f"\n=== Starting to process dataset: {dataset_name} ===")
    print("Step 1: Loading validation dataset...")
    eval_data = eval_datasets[dataset_name]  # retrieve validation dataset directly

    print("Step 2: Creating DataLoader...")
    # randomly select sample indices
    if args.traj_num == 0: # use all samples
        indices = np.arange(len(eval_data))
    else:
        indices = random.sample(range(len(eval_data)), min(args.traj_num,len(eval_data)))#*2
    # create dataset subset
    subset = Subset(eval_data, indices)
    dataloader = DataLoader(subset, batch_size=yaml_config['eval_batch_size'], collate_fn=collate_fn_infer_traj_vq)
    
    print("Step 3: Converting DataLoader to list...")
    data_list = list(dataloader)
    
    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = len(data_list) // num_gpus
    print(f"Step 4: Preparing for parallel inference using {num_gpus} GPUs")
    print(f"       Batches per GPU: {batch_size_per_gpu}")
    
    print("Step 5: Setting up output directories...")
    comment_dir = os.path.join(yaml_config['output_dir'], yaml_config['comment'])
    if not os.path.exists(comment_dir):
        os.makedirs(comment_dir)
    
    dataset_savepath = comment_dir
    if not os.path.exists(dataset_savepath):
        os.makedirs(dataset_savepath)
    print(f"       Output will be saved to: {dataset_savepath}")
    
    print("Step 6: Running parallel inference...")
    max_new_tokens = yaml_config.get('max_new_tokens', 300)
    results = parallel_infer(model, data_list, num_gpus, batch_size_per_gpu, dataset_savepath, args.save_name, \
                             max_new_tokens=max_new_tokens)

    print(f"=== Finished processing dataset: {dataset_name} ===\n")

# --------------------- Callbacks & evaluation ---------------------

class CustomEvalAndLogCallback(TrainerCallback):
    def __init__(self, eval_datasets, sample_size=None, eval_step_per=100):
        self.eval_datasets = eval_datasets
        self.sample_size = sample_size
        self.eval_step_per = eval_step_per

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_step_per == 0:
            trainer = self.trainer
            log_data = {"global_step": state.global_step}

            for ds_name, dataset_dict in self.eval_datasets.items():
                for split, dataset in dataset_dict.items():
                    if self.sample_size is not None:
                        indices = random.sample(range(len(dataset)), self.sample_size)
                        subset = Subset(dataset, indices)
                    else:
                        subset = dataset
                    metrics = trainer.evaluate(eval_dataset=subset)

                    log_data.update({
                        f"{ds_name}/{split}/{metric}": value
                        for metric, value in metrics.items()
                    })

            if is_main_process(local_rank=trainer.args.local_rank):  # only log to W&B from the main process
                wandb.log(log_data)
            print("Evaluation Logs:", log_data)

if __name__ == "__main__":
    # set multiprocessing start method to 'spawn' to avoid CUDA init conflicts
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="MixMulan Trainer")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--infer_type",
        type=str,
        required=True,
        help="perturb or trajectory. perturb generate next cell (including next velocity or cell after perturb), trajectory generate trajectory."
    )
    parser.add_argument(
        "--traj_num",
        type=int,
        default=10,
        required=False,
        help="Number of trajectories to generate. when infer_type==perturb, this is the testset batch number."
    )

    args = parser.parse_args()
    yaml_path = args.config_file
    yaml_config = load_yaml_config(yaml_path)

    # simplified experiment record
    experiment_data = {
        "comment": yaml_config.get("comment", ""),
        "dataset_names": yaml_config.get("dataset_names", []),  # record all dataset names directly
        "ckpt_path": yaml_config.get("ckpt_path", "")
    }

    # retrieve config items; support both string and list formats
    data_folders = yaml_config.get("data_folders", "")
    data_types = yaml_config.get("data_types", "")

    # convert string to single-element list
    if isinstance(data_folders, str):
        data_folders = [data_folders]
    if isinstance(data_types, str):
        data_types = [data_types]

    # ensure data_folders and data_types each have at least one element
    if len(data_folders) == 0:
        raise ValueError("data_folders cannot be empty")
    if len(data_types) == 0:
        raise ValueError("data_types cannot be empty")


    # initialize each dataset using initialize_datasets_from_config
    eval_datasets = {}
    for ds_name in yaml_config["dataset_names"]:
        # build config for each dataset
        dataset_config = {
            "data_folders": data_folders,  # shared folder config
            "dataset_names": [ds_name],    # only the current dataset name
            "meta_info_name": yaml_config["meta_info_name"],
            "block_size": yaml_config["block_size"],
            "mapping_dict": yaml_config["mapping_dict"],
            "global_dataset": yaml_config["global_dataset"],
            "data_types": data_types,      # shared data type config
            "vq_vae_path": yaml_config.get("vq_vae_path",'/hpc-cache-pfs/home/bianhaiyang/veloMulan/outputHub/vqvae_ckpt/cvqvae_scbasecount_fixed_recon1e4/checkpoint-200000/vqmodel'),
            # trajectory_perturb_h5ad specific
            "perturb_config": yaml_config.get("perturb_config", None),
            "trajectory_pkl": yaml_config.get("trajectory_pkl", None),
        }
        
        # trajectory: run on test set; trajectory_h5ad: generate from h5ad file;
        # trajectory_perturb_h5ad: perturb intermediate cells and generate continuations
        if args.infer_type == 'trajectory_scbasetraj':
            _, datasets_dict = initialize_datasets_from_config(dataset_config, skip_train=True)
        elif args.infer_type == 'trajectory_h5ad':
            _, datasets_dict = initialize_datasets_from_config_h5ad(dataset_config, skip_train=True)
        elif args.infer_type == 'trajectory_perturb_h5ad':
            _, datasets_dict = initialize_datasets_from_config_h5ad_traj(dataset_config, skip_train=True)
        else:
            raise ValueError('infer_type not supported')
        
        # add current dataset to eval_datasets
        eval_datasets[ds_name] = datasets_dict[ds_name]

    eval_cell_num = yaml_config['eval_cell_num']
    random.seed(42)

    # get vocab_size and load model
    # retrieve vocab_size from the first dataset
    first_dataset_name = yaml_config['dataset_names'][0]
    vocab_size = eval_datasets[first_dataset_name].vocab_size
    model_config = get_model_config_from_yaml(yaml_config, vocab_size)
    model_ckpt = yaml_config['ckpt_path']
    model = CellTempo_backbone.from_pretrained(model_ckpt, config=model_config)
    model.eval()

    # set log path
    log_path = "./experiments_log.json"
    setattr(args, 'save_name', yaml_config['save_name'])

    # process each dataset
    for dataset_name in yaml_config['dataset_names']:
        process_dataset(dataset_name, args)

    # append experiment record after all datasets are processed
    append_to_json_file(experiment_data, log_path)
    print(f"Experiment data for all datasets appended to {log_path}")
    print(f"All datasets processed successfully!")
