## CellTempo: Autoregressive forecasting of single-cell state transitions
CellTempo is a temporal single-cell AI model that learns cellular dynamics as a generative process and can forecast long-range cell-state transition trajectories from snapshot measurements in an autoregressive manner. It first compresses single-cell transcriptomes into compact discrete codes, and then models temporal trajectories autoregressively using ordered sequences of discrete cell codes. 

To train this model, we constructed scBaseTraj, a single-cell temporal trajectory dataset that integrates RNA velocity, pseudotime, and inferred transition probabilities to compose biologically grounded multi-step cellular sequences. Trained on scBaseTraj, CellTempo can forecast cell state evolutions from individual cells, reconstruct cellular potential landscapes of preferred directions and tendencies of cell state progression, and predict both short-term and long-term cellular responses of perturbations. The scBaseTraj dataset can be found at Hugging Face: EperLuo/scBaseTraj

We evaluated CellTempo across multiple human single-cell studies and found that long-range trajectories generated from static transcriptomic snapshots faithfully recapitulate transcriptional dynamics, lineage topology, and cell-type composition observed in real systems. We used CellTempo to reconstruct the cellular potential landscape of human hematopoiesis, recovering known differentiation hierarchies, and further showed that perturbation of lineage-associated gene modules redirects generated trajectories toward biologically consistent differentiation fates within the landscape. Extending this framework to chemical perturbations, CellTempo propagates ATRA-induced perturbations in hematopoietic stem cells forward in time predict differentiation trajectories biased toward specific lineage branches and reproduces immediate and delayed drug responses of breast and colorectal cancer cell lines.

## Repository Structure
- configs/ — configuration files for model, training, and inference
- data/ — training and inference datasets
- notebooks/ — scripts for analysis and evaluation of generation results. Reproduce the main results in the paper.
- outputs/ — generated results and models checkpoint
- scripts/ — scripts for training and inference
- src/ — source code for model, trainer, and utils

## Environment & Dependencies
```bash
conda create -n celltempo python=3.10
conda activate celltempo
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
Then install cytotrace2: 
```
git clone https://github.com/digitalcytometry/cytotrace2
cd cytotrace2/cytotrace2_python
pip install -e .
```
Note: there may be some issues related to the version of scib package. Please install from source if you encounter any problems: https://github.com/theislab/scib.

## Data Preparation
We suggest to place the raw data in the `data/` directory.
The full scBasetraj dataset can be found at:
Other example datasets we used in the manuscript can be found at:

## Training
Note: modify the paths in the script to match your local directory before running the following steps.

Step1: Train the cell VQVAE model
```bash
sh scripts/trainer_celltempo_vqvae.sh
```

Step2: Train the cell autoregressive model
```bash
sh scripts/trainer_celltempo_backbone.sh
```

## Inference & Generation
Note again: modify the paths in the script to match your local directory before running the following steps.
We support the following inference types:
- `trajectory_scbasetraj`: generate trajectory from the test set of scBasetraj dataset.
- `trajectory_h5ad`: generate trajectory from a given h5ad file.
- `trajectory_perturb_h5ad`: perturb intermediate cells in trajectories generated from a given h5ad file.
For the last one, you need to construct trajectories and specify the perturbed cell id. You can refer to the `notebooks/evaluation_landscape.ipynb` file for more details.

examples can be found in the `scripts/generator_traj.sh` file.
```
sh scripts/generator_traj.sh
```

## Evaluation & Analysis
For the evaluation of `trajectory_scbasetraj`, please refer to the `notebooks/evaluation_scbasetraj_testset.ipynb` file.
For the evaluation of `trajectory_h5ad`, please refer to the `notebooks/evaluation_h5ad_file.ipynb` file.
For the evaluation of `trajectory_perturb_h5ad`, please refer to the `notebooks/evaluation_landscape.ipynb` file.


## Citation & Acknowledgements
- Citation format
- Acknowledgements list
