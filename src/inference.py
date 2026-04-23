import torch
from torch.utils.data import DataLoader
from multiprocessing import Pool
from tqdm import tqdm
import os
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

def move_to_device(batch_data, device):
    """将batch数据移动到对应的GPU"""
    input_ids = batch_data['input_ids'].to(device)
    # x_expr = batch_data['x_expr'].to(device)
    cell_pos = batch_data['cell_pos'].to(device)
    return input_ids, cell_pos


def infer_on_device(batch_data, model, device, max_new_tokens=200):
    """在指定的GPU上进行推理"""
    try:
        input_ids, cell_pos = move_to_device(batch_data, device)

        with open(str(BASE_DIR / 'data/mix_meta_info_vq_traj.json'), 'r') as f:
            meta_info = json.load(f)
        ingnore_id = []
        chars = meta_info['token_set']
        for i in range(len(chars)):
            if chars[i][0] not in [str(i) for i in range(10)]:
                ingnore_id.append(i)   
        # 推理
        generated_input_ids, generated_x_expr, generated_cell_pos = model.generate_debug(
            input_ids=input_ids,
            ignore_Idx = ingnore_id,
            cell_pos=cell_pos,
            max_new_tokens=max_new_tokens,
            top_k=1,
            use_cache=True,
            debug=False,
        )
        
        # 将每条数据的推理结果存成字典
        results = []
        for i in range(len(batch_data['input_ids'])):
            result = {
                'generated_ids': generated_input_ids[i].cpu().tolist(),  # 转到 CPU
                'entropy': generated_x_expr[i].cpu().tolist(),    # 转到 CPU
                'token_labels': batch_data['token_labels'][i].cpu().tolist(),  # 转到 CPU
                'expr_labels': batch_data['input_ids'][i].cpu().tolist(),    # 转到 CPU
                'idx': batch_data['idx'][i],
            }
            results.append(result)
        
        return results
    except RuntimeError as e:
        # 捕获CUDA错误并打印详细信息
        print(f"CUDA错误发生在处理批次时: {str(e)}")
        print(f"批次信息: 输入ID形状 = {batch_data['input_ids'].shape}, device = {device}")
        # 返回空结果而不是崩溃
        return []


def process_batch_on_device(gpu_data):
    """每个GPU执行的任务，保存推理结果到指定路径"""
    device, batch_data, model, savepath, save_name, gpu_id, max_new_tokens = gpu_data
    model = model.to(device)
    
    results = []
    total_batches = len(batch_data)
    print(f"\nGPU {gpu_id}: Starting to process {total_batches} batches")
    
    # 修改tqdm的配置
    pbar = tqdm(
        batch_data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,  # 为每个GPU设置固定的进度条位置
        leave=True,      # 保持进度条显示
        ncols=100,       # 设置进度条宽度
        unit='batch'     # 显示处理单位
    )
    
    # 处理该GPU上分配的batch list中的每个batch
    for i, batch in enumerate(pbar):
        result = infer_on_device(batch, model, device, max_new_tokens=max_new_tokens)
        results.extend(result)

        # 更新进度条描述
        pbar.set_description(f"GPU {gpu_id}: {i+1}/{total_batches} batches")

    # 将结果保存为 JSON 文件
    save_file = os.path.join(savepath, f"gpu_{gpu_id}_results_{save_name}.pt")
    torch.save(results, save_file)

    print(f"\nGPU {gpu_id}: Saving results to {save_file}")
    
    print(f"GPU {gpu_id}: Processing completed")
    return f"Results for GPU {gpu_id} saved to {save_file}"

def parallel_infer(model, data_list, num_gpus, batch_size_per_gpu, savepath, save_name, max_new_tokens=300):
    """在多张GPU卡上并行推理，并保存结果"""
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]  # 获取所有GPU设备
    
    # 把batch list分成N份，每份分配给不同的GPU
    data_splits = [data_list[i:i + batch_size_per_gpu] for i in range(0, len(data_list), batch_size_per_gpu)]
    
    # 使用multiprocessing的Pool进行并行化
    with Pool(processes=num_gpus) as pool:
        # 每个GPU处理一部分数据，并保存结果
        gpu_data = [(devices[i], data_splits[i], model, savepath, save_name, i, max_new_tokens) for i in range(len(devices))]
        results = pool.map(process_batch_on_device, gpu_data)

    # 返回所有GPU的保存信息
    return results
