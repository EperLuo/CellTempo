import torch
from torch.utils.data import DataLoader
from multiprocessing import Pool
from tqdm import tqdm
import os
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

def move_to_device(batch_data, device):
    """Move a batch of data to the specified GPU device."""
    input_ids = batch_data['input_ids'].to(device)
    # x_expr = batch_data['x_expr'].to(device)
    cell_pos = batch_data['cell_pos'].to(device)
    return input_ids, cell_pos


def infer_on_device(batch_data, model, device, max_new_tokens=200):
    """Run inference on the specified GPU."""
    try:
        input_ids, cell_pos = move_to_device(batch_data, device)

        with open(str(BASE_DIR / 'data/mix_meta_info_vq_traj.json'), 'r') as f:
            meta_info = json.load(f)
        ingnore_id = []
        chars = meta_info['token_set']
        for i in range(len(chars)):
            if chars[i][0] not in [str(i) for i in range(10)]:
                ingnore_id.append(i)   
        # inference
        generated_input_ids, generated_x_expr, generated_cell_pos = model.generate_debug(
            input_ids=input_ids,
            ignore_Idx = ingnore_id,
            cell_pos=cell_pos,
            max_new_tokens=max_new_tokens,
            top_k=1,
            use_cache=True,
            debug=False,
        )
        
        # store inference results for each sample as a dict
        results = []
        for i in range(len(batch_data['input_ids'])):
            result = {
                'generated_ids': generated_input_ids[i].cpu().tolist(),  # move to CPU
                'entropy': generated_x_expr[i].cpu().tolist(),    # move to CPU
                'token_labels': batch_data['token_labels'][i].cpu().tolist(),  # move to CPU
                'expr_labels': batch_data['input_ids'][i].cpu().tolist(),    # move to CPU
                'idx': batch_data['idx'][i],
            }
            results.append(result)
        
        return results
    except RuntimeError as e:
        # catch CUDA errors and print details
        print(f"CUDA error during batch processing: {str(e)}")
        print(f"Batch info: input_ids shape = {batch_data['input_ids'].shape}, device = {device}")
        # return empty results instead of crashing
        return []


def process_batch_on_device(gpu_data):
    """Task executed on each GPU; saves inference results to the specified path."""
    device, batch_data, model, savepath, save_name, gpu_id, max_new_tokens = gpu_data
    model = model.to(device)
    
    results = []
    total_batches = len(batch_data)
    print(f"\nGPU {gpu_id}: Starting to process {total_batches} batches")
    
    # configure tqdm progress bar
    pbar = tqdm(
        batch_data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,  # fixed position per GPU
        leave=True,       # keep progress bar visible after completion
        ncols=100,        # progress bar width
        unit='batch'      # unit label
    )
    
    # process each batch assigned to this GPU
    for i, batch in enumerate(pbar):
        result = infer_on_device(batch, model, device, max_new_tokens=max_new_tokens)
        results.extend(result)

        # update progress bar description
        pbar.set_description(f"GPU {gpu_id}: {i+1}/{total_batches} batches")

    # save results to file
    save_file = os.path.join(savepath, f"gpu_{gpu_id}_results_{save_name}.pt")
    torch.save(results, save_file)

    print(f"\nGPU {gpu_id}: Saving results to {save_file}")
    
    print(f"GPU {gpu_id}: Processing completed")
    return f"Results for GPU {gpu_id} saved to {save_file}"

def parallel_infer(model, data_list, num_gpus, batch_size_per_gpu, savepath, save_name, max_new_tokens=300):
    """Run parallel inference across multiple GPUs and save the results."""
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]  # get all GPU devices
    
    # split the batch list into N parts, one per GPU
    data_splits = [data_list[i:i + batch_size_per_gpu] for i in range(0, len(data_list), batch_size_per_gpu)]
    
    # use multiprocessing Pool for parallelism
    with Pool(processes=num_gpus) as pool:
        # each GPU processes a subset of the data and saves results
        gpu_data = [(devices[i], data_splits[i], model, savepath, save_name, i, max_new_tokens) for i in range(len(devices))]
        results = pool.map(process_batch_on_device, gpu_data)

    # return save info from all GPUs
    return results
