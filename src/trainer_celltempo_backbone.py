import wandb
import json

import os
import sys
import yaml
import random
import torch
from torch.utils.data import Subset
from transformers.trainer_utils import is_main_process
from transformers import Trainer, TrainingArguments, TrainerCallback
import argparse

# set project root path
root_path = os.path.abspath('/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/mixMulan_AR_traj/')
sys.path.append(root_path)

# import model and data modules
from utils.train_utils import initialize_datasets_from_config, get_dataset_config_from_yaml, initialize_datasets_from_config_perturb
from model.CellTempo_backbone import CellTempo_backbone, CellTempoConfig
from utils.dataset import collate_fn_train_traj_vq

os.environ["HF_HOME"] = "/voyager-data/luoerpai/hf_cache/"
os.environ["HF_DATASETS_CACHE"]="/voyager-data/luoerpai/hf_cache/"

os.environ["PYARROW_NUM_THREADS"] = "112"     # or set a reasonable core count manually
os.environ["OMP_NUM_THREADS"] = "112"

# --------------------- Config loading & processing ---------------------

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
        meta_info_name=yaml_config["meta_info_name"]
    )

class CellTempoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        rewrite the compute_loss method to add the detailed loss components to the evaluation metrics
        """
        # call standard loss computation
        outputs = model(**inputs)
        loss = outputs.loss

        # temporarily store sub-losses for this step (detached as float)
        if model.training:
            logs = {}
            for k in ["loss_cls_postfix", "loss_exp_postfix"]:
                if hasattr(outputs, k) and getattr(outputs, k) is not None:
                    v = getattr(outputs, k)
                    logs[f"train/{k}"] = float(v.detach().mean().item())
            # store on the trainer instance; callbacks read after optimizer.step
            if logs:
                self._last_losses = logs

        # in eval mode, add detailed loss components to eval metrics
        if not model.training and hasattr(outputs, "loss_cls_postfix") and hasattr(outputs, "loss_exp_postfix"):
            # add these values to eval metrics
            if not hasattr(self, "eval_metrics"):
                self.eval_metrics = {}
            
            # store detailed loss components
            if outputs.loss_cls_postfix is not None:
                self.eval_metrics["eval_loss_cls_postfix"] = outputs.loss_cls_postfix.detach().item()
            
            if outputs.loss_exp_postfix is not None:
                self.eval_metrics["eval_loss_exp_postfix"] = outputs.loss_exp_postfix.detach().item()
        
        return (loss, outputs) if return_outputs else loss
        
    def evaluation_loop(self, *args, **kwargs):
        # reset eval metrics
        self.eval_metrics = {}
        
        # call the original evaluation loop
        output = super().evaluation_loop(*args, **kwargs)
        
        # append our extra metrics to the output
        if hasattr(self, "eval_metrics") and self.eval_metrics:
            output.metrics.update(self.eval_metrics)
        
        return output
    
# --------------------- Callbacks & evaluation ---------------------
class CustomEvalAndLogCallback(TrainerCallback):
    def __init__(self, eval_datasets, sample_size=None, eval_step_per=100):
        self.eval_datasets = eval_datasets
        self.sample_size = sample_size
        self.eval_step_per = eval_step_per

    def on_optimizer_step(self, args, state, control, **kwargs):
        # trainer: Trainer = kwargs["trainer"]
        trainer = self.trainer

        # only write log when Trainer decides it should (matches logging_steps/strategy)
        if hasattr(trainer, "_last_losses") and trainer._last_losses and (trainer.state.global_step % trainer.state.logging_steps == 0):
            trainer.log(trainer._last_losses)  # send to report_to (wandb/tensorboard) only, not CLI
            trainer._last_losses = {}          # clear cache to avoid duplicate logging

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_step_per == 0 or state.global_step == 0:
            trainer = self.trainer
            log_data = {"global_step": state.global_step}

            # evaluate regular datasets
            for ds_name, dataset_dict in self.eval_datasets.items():
                for split, dataset in dataset_dict.items():
                    if self.sample_size is not None:
                        indices = random.sample(range(len(dataset)), min(self.sample_size, len(dataset)))
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


# --------------------- Main training logic ---------------------

parser = argparse.ArgumentParser(description="MixMulan Trainer")
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to the YAML configuration file"
)
args = parser.parse_args()

yaml_path = args.config_file
yaml_config = load_yaml_config(yaml_path)

experiment_data = {
    "yaml_path": os.path.basename(yaml_path),
    "comment": yaml_config.get("comment", ""),
    "ckpt_path": yaml_config.get("ckpt_path", "")
}

if 'trajectory' in yaml_config.get("data_types", ["trajectory"]):
    train_dataset, eval_datasets = initialize_datasets_from_config(get_dataset_config_from_yaml(yaml_config))
elif 'perturb' in yaml_config.get("data_types", ["perturb"]):
    train_dataset, eval_datasets = initialize_datasets_from_config_perturb(get_dataset_config_from_yaml(yaml_config))
else:
    raise ValueError("data_types must contain either 'trajectory' or 'perturb'")

eval_cell_num = yaml_config['eval_cell_num']
random.seed(43)

eval_subsets = {}
for dataset_name in yaml_config['dataset_names']:
    dataset_val = eval_datasets[dataset_name]['val']
    if eval_cell_num >= len(dataset_val):
        eval_subsets[dataset_name] = {'val': dataset_val}
        continue
    indices = random.sample(range(len(dataset_val)), eval_cell_num)
    subset = Subset(dataset_val, indices)
    eval_subsets[dataset_name] = {'val': subset}

vocab_size = train_dataset.vocab_size
model_config = get_model_config_from_yaml(yaml_config, vocab_size)

if yaml_config['init_from'] == 'resume':
    ckpt_path = yaml_config['ckpt_path']
    print(f'resume from {ckpt_path}')
    model = CellTempo_backbone.from_pretrained(ckpt_path,config=model_config, ignore_mismatched_sizes=True)
else:
    model = CellTempo_backbone(model_config)

device_num = torch.cuda.device_count()

save_path = os.path.join(yaml_config['output_dir'], yaml_config['comment'])

# print(f'device_number:~~~~~~:::{device_num}')

training_args = TrainingArguments(
    
    seed=19491002,
    
    output_dir= save_path,
   
    evaluation_strategy = "no",
    eval_steps = yaml_config['eval_itervals'],
    
    num_train_epochs=yaml_config['max_epochs'],
   
    per_device_train_batch_size=yaml_config['batch_size'],
    gradient_accumulation_steps = yaml_config['gradient_accumulation_steps'],
    warmup_steps=yaml_config.get('warmup_iters', 0),
    learning_rate = yaml_config['learning_rate'],
    weight_decay = yaml_config['weight_decay'],
    lr_scheduler_type="linear",
    max_grad_norm = yaml_config['grad_clip'],
    
    save_strategy="steps", 
    save_steps=yaml_config['save_ckpt_iter'],
    save_total_limit=10,

    dataloader_num_workers=12,
    dataloader_pin_memory=False,
    
    logging_dir = os.path.join(save_path, 'logs'),
    logging_steps=yaml_config['log_interval'],
    run_name=yaml_config['comment'],
    
    ddp_backend = yaml_config['backend'],
    bf16=True,
    no_cuda=False,  # ensure CUDA is used
    remove_unused_columns=False,  # disable automatic column removal
    ddp_find_unused_parameters=True,
)

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    os.environ["WANDB_PROJECT"] = yaml_config['wandb_project']
    wandb.init(
        project=yaml_config['wandb_project'],
        name=f"{yaml_config['comment']}_train",
        config=training_args.to_dict()
    )

callback = CustomEvalAndLogCallback(
    eval_subsets, 
    eval_step_per=yaml_config['eval_itervals']
)

# use custom Trainer instead of the standard one
trainer = CellTempoTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # evaluation handled by custom callback
    callbacks=[callback],
    data_collator=collate_fn_train_traj_vq,
)

# set trainer reference so the callback can access it
callback.trainer = trainer

if yaml_config['init_from'] == 'resume':
    trainer.train(resume_from_checkpoint=yaml_config['ckpt_path'])
else:
    trainer.train()
