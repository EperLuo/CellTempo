export "CUDA_DEVICE_ORDER=PCI_BUS_ID"

# alias proxy-on="export http_proxy=http://100.68.163.252:3128 https_proxy=http://100.68.163.252:3128 HTTP_PROXY=http://100.68.163.252:3128 HTTPS_PROXY=http://100.68.163.252:3128"
# alias proxy-off="unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY"
# proxy-on

export HF_HOME="/data/lep/hf_cache/"
export HF_DATASETS_CACHE="/data/lep/hf_cache/"
: "${WANDB_API_KEY:?Please set WANDB_API_KEY in your environment before running this script}"
export WANDB_API_KEY

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node ${MLP_WORKER_GPU:-2} \
    --master_addr ${MLP_WORKER_0_HOST:-127.0.0.1} \
    --node_rank ${MLP_ROLE_INDEX:-0} \
    --master_port 20002 \
    --nnodes ${MLP_WORKER_NUM:-1} \
    src/trainer_celltempo_backbone.py \
    --config_file "/data/lep/CellTempo_noempty/CellTempo/configs/tahoe100m_finetune.yaml"