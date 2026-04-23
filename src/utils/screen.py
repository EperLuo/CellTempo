import torch

@torch.no_grad()
def estimate_loss(eval_datasets, cfg, model, device, ctx):

    """
    Compute training and validation losses for multiple datasets.

    Args:
        eval_datasets (dict): Evaluation dataset information for all datasets.
        cfg (object): Config object containing parameters such as `eval_iters`.
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): Device to run evaluation on.
        ctx (context manager): Context manager, e.g. `torch.cuda.amp.autocast()`.

    Returns:
        dict: Training and validation losses for each dataset.
    """
    out = {}
    model.eval()

    for ds_name, splits in eval_datasets.items():
        out[ds_name] = {}
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters, device=device)
            data_loader = splits[split]['loader']
            data_iter = splits[split]['iter']

            for iteration in range(cfg.eval_iters):
                try:
                    data_one = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    data_one = next(data_iter)
                # update iterator
                eval_datasets[ds_name][split]['iter'] = data_iter

                # data loading
                idx = data_one['tokens'].pin_memory().to(device, non_blocking=True)
                targets = data_one['labels_tokens'].pin_memory().to(device, non_blocking=True)
                xlen = data_one['data_len'].pin_memory().to(device, non_blocking=True)
                c1_len = data_one['c1_len'].pin_memory().to(device, non_blocking=True)
                c2_start = data_one['c2_start'].pin_memory().to(device, non_blocking=True)
                x_expr = data_one['values'].pin_memory().to(device, non_blocking=True)
                y_expr = data_one['labels_values'].pin_memory().to(device, non_blocking=True)
                cell_pos = data_one['cell_pos'].pin_memory().to(device, non_blocking=True)

                with ctx:
                    _, _, loss, _, _, _, _ = model(
                        idx=idx,
                        targets=targets,
                        xlen=xlen,
                        c1_len=c1_len,
                        c2_start=c2_start,
                        x_expr=x_expr,
                        y_expr=y_expr,
                        cell_pos=cell_pos
                    )

                if isinstance(loss, (float, int)):
                    losses[iteration] = loss
                else:
                    losses[iteration] = loss.item()

            out[ds_name][split] = losses.mean().item()

    model.train()
    return out




def run_evaluation(
    epoch,
    iter_num,
    eval_datasets,
    cfg,
    model,
    device,
    ctx,
    dataset_names,
    logger,
    wandb_log=False,
    wandb=None,
    all_eval_loss=None,
    all_eval_iter=None,
    lr=None,
    running_mfu=None
):
    """
    Run an evaluation step, computing training and validation losses for multiple
    datasets, and log the results.

    Args:
        epoch (int): Current epoch index.
        iter_num (int): Current iteration step.
        eval_datasets (dict): Evaluation dataset information for all datasets.
        cfg (object): Config object containing parameters such as `eval_iters`.
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): Device to run evaluation on.
        ctx (context manager): Context manager, e.g. `torch.cuda.amp.autocast()`.
        dataset_names (list): List of dataset names.
        logger (logging.Logger): Logger instance.
        wandb_log (bool): Whether to enable W&B logging.
        wandb (object): W&B instance.
        all_eval_loss (dict): Storage for evaluation losses across all datasets.
        all_eval_iter (dict): Storage for evaluation iteration steps across all datasets.
        lr (float): Current learning rate.
        running_mfu (float): Current MFU (Model FLOPs Utilization).

    Returns:
        None
    """
    
    # compute losses
    losses = estimate_loss(eval_datasets, cfg, model, device, ctx)
    
    # log losses for each dataset
    for ds_name in dataset_names:
        train_loss = losses[ds_name]['train']
        val_loss = losses[ds_name]['val']
        logger.info(f"Dataset: {ds_name} | Step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        
        if wandb_log and wandb is not None:
            wandb.log({
                'epoch': epoch,
                'step': iter_num,
                f'{ds_name}/train_loss': train_loss,
                f'{ds_name}/val_loss': val_loss,
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        
        # update evaluation result storage
        all_eval_iter[ds_name].append(iter_num)
        all_eval_loss[ds_name]['train'].append(train_loss)
        all_eval_loss[ds_name]['val'].append(val_loss)





import os
import torch

def save_checkpoint(
    iter_num,
    raw_model,
    optimizer,
    model_args,
    best_val_loss,
    all_eval_iter,
    all_eval_loss,
    ckpt_save_dir,
    logger,
):
    """
    Save a training checkpoint.

    Args:
        iter_num (int): Current iteration step.
        cfg (object): Config object containing parameters such as `always_save_checkpoint`
            and `save_ckpt_iter`.
        raw_model (torch.nn.Module): Raw model (not wrapped in a distributed container).
        optimizer (torch.optim.Optimizer): Optimizer instance.
        model_args (dict): Model initialization arguments or other relevant parameters.
        best_val_loss (float): Best validation loss seen so far.
        all_eval_iter (dict): Evaluation iteration steps stored per dataset.
        all_eval_loss (dict): Training and validation losses stored per dataset.
        ckpt_save_dir (str): Directory to save the checkpoint.
        logger (logging.Logger): Logger instance.
        master_process (bool): Whether this is the main process (controls logging and
            saving in distributed training).

    Returns:
        None
    """
    # build checkpoint dict
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'eval_iter': all_eval_iter,
        'train_loss': {ds: losses['train'] for ds, losses in all_eval_loss.items()},
        'val_loss': {ds: losses['val'] for ds, losses in all_eval_loss.items()},
    }
    
    # build checkpoint file path
    ckpt_filename = f'ckpt{iter_num}.pt'
    ckpt_path = os.path.join(ckpt_save_dir, ckpt_filename)
    logger.info(f"Saving checkpoint to {ckpt_path}")
    
    # save checkpoint
    torch.save(checkpoint, ckpt_path)

