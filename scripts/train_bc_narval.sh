#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G      
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-dpmeger

module load StdEnv/2020 gcc/9.3.0
module load opencv/4.8.0
module load python/3.9
module load apptainer
source /home/jftrem/projects/def-dpmeger/jftrem/topological_mapping_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/jftrem/projects/def-dpmeger/jftrem/topological_mapping/src/
python /home/jftrem/projects/def-dpmeger/jftrem/topological_mapping/src/topological_mapping/learning_bc/trainer.py \
    hist_size=5 \
    only_front=False \
    version=s \
    batch_size=300 \
    learning_rate=0.00005 \
    DINO=True \
    past_actions=True \
    velocities=True \
    hidden_layers=4 \
    hidden_dim=4096 \
    residual=False \
    gelu=False \
    mirror_trajs=False \
    tiny_dataset=False \
    n_epochs=40 \
    paths=narval \
    distributed=True \
    world_size=1 \
    n_simulations=20 \
    num_workers=4
