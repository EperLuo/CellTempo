export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

cd src

# Run the whole test set of scBasetraj
python generate_traj.py --config_file /hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/CellTempo/configs/generate_traj_scBasetraj_testset.yaml --infer_type trajectory_scbasetraj --traj_num 100

# # Generate trajectory from h5ad file
# python generate_traj.py --config_file /hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/CellTempo/configs/generate_traj_h5ad_file.yaml --infer_type trajectory_h5ad --traj_num 0

# # Perturb intermediate cells in trajectory
# python generate_traj.py --config_file /hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/CellTempo/configs/generate_traj_h5ad_file.yaml --infer_type trajectory_perturb_h5ad --traj_num 0
