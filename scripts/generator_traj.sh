export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_VISIBLE_DEVICES=1,2,3

cd src

# # Run the whole test set of scBasetraj
# python generate_traj.py --config_file /hpc-cache-pfs/home/bianhaiyang/veloMulan/codeHub/CellTempo/configs/generate_traj_scBasetraj_testset.yaml --infer_type trajectory_scbasetraj --traj_num 100

# # Generate trajectory from h5ad file
# python generate_traj.py --config_file /data/lep/CellTempo_noempty/CellTempo/configs/generate_traj_h5ad_file.yaml --infer_type trajectory_h5ad --traj_num 0

# Perturb intermediate cells genes in trajectory
# Requires two extra fields in the YAML config:
#   perturb_config: path to configs/perturb_gene_config.yaml  (gene modules + amplify rules)
#   trajectory_pkl: path to the .pkl file with (trajectory_list, target_id)
# bonemarrow gene perturbation exp
python generate_traj.py \
    --config_file /data/lep/CellTempo_noempty/CellTempo/configs/generate_traj_h5ad_file.yaml \
    --infer_type trajectory_perturb_h5ad \
    --traj_num 0

# OPC gene perturbation exp
python generate_traj.py \
    --config_file /data/lep/CellTempo_noempty/CellTempo/configs/generate_traj_iPSC.yaml \
    --infer_type trajectory_perturb_h5ad \
    --traj_num 0

