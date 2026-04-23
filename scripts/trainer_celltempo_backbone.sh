export "CUDA_DEVICE_ORDER=PCI_BUS_ID"

alias proxy-on="export http_proxy=http://100.68.163.252:3128 https_proxy=http://100.68.163.252:3128 HTTP_PROXY=http://100.68.163.252:3128 HTTPS_PROXY=http://100.68.163.252:3128"
alias proxy-off="unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY"
proxy-on

export HF_HOME="/voyager-data/luoerpai/hf_cache/"
export HF_DATASETS_CACHE="/voyager-data/luoerpai/hf_cache/"
export WANDB_API_KEY="f14729fe2acf6d7450dc1603f8c31f84cb4a104e"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

torchrun \
    --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port 20000 \
    --nnodes $MLP_WORKER_NUM \
    src/trainer_celltempo_backbone.py \
    --config_file "/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/CellTempo/configs/celltempo_scbasetraj_pretrain.yaml"