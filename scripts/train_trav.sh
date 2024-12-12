#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G      
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-dpmeger

module load StdEnv/2020 gcc/9.3.0
module load opencv/4.8.0
module load python/3.9
source /home/jftrem/projects/def-dpmeger/jftrem/topological_mapping_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/jftrem/projects/def-dpmeger/jftrem/topological_mapping/src/
python /home/juliea/projects/def-dpmeger/juliea/topological_mapping/src/topological_mapping/learning_trav/trainer.py  --multirun version=s     batch_size=32     n_layers=4     hidden_dim=1024     num_workers=8