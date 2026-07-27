: "${WANDB_API_KEY:?Please set WANDB_API_KEY in your environment before running this script}"
export WANDB_API_KEY
export CUDA_VISIBLE_DEVICES=2

python -u src/trainer_celltempo_vqvae.py \
  --mixed_precision=bf16 \
  --train_data_dir=/data/lep/CellTempo_noempty/CellTempo/data/Tahoe100m \
  --output_dir=/data/lep/CellTempo_noempty/ckpt/model_ckpt/vqvae_scratch_tahoe \
  --resolution=128 \
  --log_steps=500 \
  --checkpointing_steps=10000 \
  --checkpoints_total_limit=5 \
  --train_batch_size=256 \
  --max_train_steps=400000 \
  --lr_warmup_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler=linear \
  --gradient_accumulation_steps=1 \
  --report_to=wandb \
  --tracker_project_name=celltempo \
  --run_name=vqvae_scratch_tahoe \
  --dataloader_num_workers=16 \
  --data_type=rna \
  --num_gene=18791 \
  --allow_tf32 \
  --vae_loss=nb \
  --data_source=tahoe \
  --vq

  # --pretrained_model_name_or_path=/data/lep/CellTempo_noempty/ckpt/model_ckpt/vqvae_train_on_scbasecount/checkpoint-700000/vqmodel \