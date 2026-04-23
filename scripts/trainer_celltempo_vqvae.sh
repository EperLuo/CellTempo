alias proxy-on="export http_proxy=http://100.68.163.252:3128 https_proxy=http://100.68.163.252:3128 HTTP_PROXY=http://100.68.163.252:3128 HTTPS_PROXY=http://100.68.163.252:3128"
alias proxy-off="unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY"
proxy-on
export WANDB_API_KEY="f14729fe2acf6d7450dc1603f8c31f84cb4a104e"


accelerate launch --main_process_port=8666 src/trainer_celltempo_vqvae.py \
  --train_data_dir=/hpc-cache-pfs/home/bianhaiyang/veloMulan/dataHub/scBaseCount_scvelo_rowcount_filtered/scBasetraj_exp \
  --output_dir=/hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/CellTempo/outputs/vqvae_ckpt/vqvae_test \
  --resolution=128 \
  --log_steps=500 \
  --checkpointing_steps=100 \
  --checkpoints_total_limit=10 \
  --train_batch_size=64 \
  --max_train_steps=1200000 \
  --lr_warmup_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler=linear \
  --gradient_accumulation_steps=1 \
  --report_to=wandb \
  --tracker_project_name=celltempo \
  --run_name=vqvae \
  --dataloader_num_workers=4 \
  --data_type=rna \
  --num_gene=18791 \
  --allow_tf32 \
  --vae_loss=nb \
  --resume_from_checkpoint=latest \
  --vq 