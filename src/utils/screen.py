import torch

@torch.no_grad()
def estimate_loss(eval_datasets, cfg, model, device, ctx):

    """
    计算多个数据集的训练和验证损失。

    参数:
        eval_datasets (dict): 包含所有评估数据集的信息。
        cfg (object): 配置对象，包含 `eval_iters` 等参数。
        model (torch.nn.Module): 评估的模型。
        device (torch.device): 设备信息。
        ctx (context manager): 上下文管理器，如 `torch.cuda.amp.autocast()`

    返回:
        dict: 包含每个数据集的训练和验证损失。
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
                # 更新迭代器
                eval_datasets[ds_name][split]['iter'] = data_iter

                # 数据处理
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
    执行评估步骤，计算多个数据集的训练和验证损失，并记录结果。

    参数:
        epoch (int): 当前的epoch编号。
        iter_num (int): 当前的迭代步数。
        eval_datasets (dict): 包含所有评估数据集的信息。
        cfg (object): 配置对象，包含 `eval_iters` 等参数。
        model (torch.nn.Module): 评估的模型。
        device (torch.device): 设备信息。
        ctx (context manager): 上下文管理器，如 `torch.cuda.amp.autocast()`
        dataset_names (list): 数据集名称列表。
        logger (logging.Logger): 日志记录器。
        wandb_log (bool): 是否启用 WandB 日志记录。
        wandb (object): WandB 实例。
        all_eval_loss (dict): 存储所有数据集的评估损失。
        all_eval_iter (dict): 存储所有数据集的评估迭代步数。
        lr (float): 当前的学习率。
        running_mfu (float): 当前的 MFU（Memory Footprint Utilization）。

    返回:
        None
    """
    
    # 调用 estimate_loss 计算损失
    losses = estimate_loss(eval_datasets, cfg, model, device, ctx)
    
    # 遍历每个数据集并记录损失
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
                "mfu": running_mfu * 100,  # 转换为百分比
            })
        
        # 更新评估结果存储字典
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
    保存训练检查点。

    参数:
        iter_num (int): 当前的迭代步数。
        cfg (object): 配置对象，包含 `always_save_checkpoint` 和 `save_ckpt_iter` 等参数。
        raw_model (torch.nn.Module): 原始模型（未封装在分布式包装器中）。
        optimizer (torch.optim.Optimizer): 优化器实例。
        model_args (dict): 模型的初始化参数或其他相关参数。
        best_val_loss (float): 当前最佳的验证损失。
        all_eval_iter (dict): 存储各数据集评估的迭代步数。
        all_eval_loss (dict): 存储各数据集的训练和验证损失。
        ckpt_save_dir (str): 检查点保存的目录。
        logger (logging.Logger): 日志记录器。
        master_process (bool): 是否为主进程（在分布式训练中用于控制日志和保存）。

    返回:
        None
    """
    # 构建检查点字典
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
    
    # 构建检查点文件路径
    ckpt_filename = f'ckpt{iter_num}.pt'
    ckpt_path = os.path.join(ckpt_save_dir, ckpt_filename)
    logger.info(f"Saving checkpoint to {ckpt_path}")
    
    # 保存检查点
    torch.save(checkpoint, ckpt_path)

