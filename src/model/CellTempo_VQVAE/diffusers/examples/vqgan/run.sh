export HF_ENDPOINT=https://hf-mirror.com
export PATH="/data1/lep/anaconda3/envs/diffusion_v2/bin:$PATH"
accelerate launch --main_process_port=8080 train_vqgan.py \
  --dataset_name=cifar10 \
  --image_column=img \
  --output_dir=/data1/lep/Workspace/scdiffusion_v2/output/VAE/vqgan_img \
  --validation_images images/bird.jpg images/car.jpg images/dog.jpg images/frog.jpg \
  --resolution=128 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=8 \
  --report_to=wandb