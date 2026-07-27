export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_VISIBLE_DEVICES=1,2,3

cd src

# The two pipeline examples below must be run from CellTempo/src (this script
# switches to that directory above).  Consequently, output paths should use
# ../output/ rather than CellTempo/output/.

# # Tahoe100m perturbation generation (control -> perturbed)
# # Uses drug UniMol embeddings; generates perturbed cells from control cells on testA split.
# # Set ckpt_path in the YAML config to the trained perturbation checkpoint.
# # --traj_num 0 means use all samples.
# python generate_traj.py \
#     --config_file /data/lep/CellTempo_noempty/CellTempo/configs/generate_perturb_tahoe_test.yaml \
#     --infer_type perturb_tahoe \
#     --traj_num 0

# # Perturbation from h5ad + single SMILES (control -> perturbed)
# # Reads all cells from the h5ad file, applies the specified drug perturbation.
# # Drug embedding is first looked up from drug_emb_paths in config;
# # if not found, computed on-the-fly via unimol_tools UniMolRepr (310M model).
# # --smiles and --h5ad_path can override the YAML config values.
# python generate_traj.py \
#     --config_file /data/lep/CellTempo_noempty/CellTempo/configs/generate_perturb_drug_h5ad.yaml \
#     --infer_type perturb_h5ad \
#     --traj_num 0 \
#     --dose dose_5.0

# Single-drug DEG trajectory pipeline.
#
# Usage: the default mode processes every drug in data/target_mole.txt.  Add
# --drug <drug_name> to process just one drug.  With --deg_traj_mode, DMSO is
# first generated as the reference; each drug's DEGs versus DMSO are then used
# to amplify the top up/down genes and generate a separate trajectory.
#
# This example keeps only GPC_prolif cells, uses a one-cell prefix at step 0
# (--prefix_mode single), and uses the top 12 up/down DEGs per drug.  The
# intermediate perturbation/trajectory/DEG files are tagged exp08 so they do
# not collide with another run.  Existing perturbation results are reused by
# default; add --no-skip_perturb to force them to be generated again.
#
# Required before running: DMSO and the requested drug names must be present
# in ../data/target_mole.txt, and the checkpoint paths in the Python script
# must be available.  The final per-drug, per-step cell-type counts are CSV.
python pipeline_drug_screen_opc.py \
    --deg_traj_mode \
    --dose dose_5.0 --cell_type GPC_prolif --prefix_mode single \
    --output ../output/deg_traj_mode_dose5_single_gpcprolif.csv \
    --cuda_devices 2,3 --run_tag exp08 \
    --deg_top_n 12

# Multi-drug union-DEG trajectory pipeline.
#
# --drug_file supplies the drugs to combine (one name per line, or
# drug_name<TAB>SMILES); DMSO is added automatically as the reference.  The
# script calculates each drug's DEGs versus DMSO, takes the union of the top
# 12 significant up/down genes from all listed drugs, then generates ONE
# combined trajectory named by --combo_name.  The names in the file must also
# be resolvable from ../data/target_mole.txt when no SMILES is supplied.
#
# This example uses the same dose/cell filter/prefix setup as the single-drug
# run.  --no_control omits the separate unperturbed control trajectory; remove
# it if a control baseline CSV is wanted.  combo_exp01 keeps generated files
# separate from other multi-drug experiments.  The resulting cell-type counts
# are written to the CSV given by --output.
python pipeline_multi_deg_traj_opc.py  \
   --drug_file /data/lep/CellTempo_noempty/CellTempo/data/drug_combo.txt  \
   --combo_name "OPC_combo_stage1" \
   --deg_top_n 12   \
   --dose dose_5.0  \
   --step_num 12  \
   --cell_type GPC_prolif  \
   --prefix_mode single   \
   --cuda_devices "0,1"  \
   --run_tag combo_exp01  \
   --output ../output/gpconly_combo_stage1.csv  \
   --no_control
