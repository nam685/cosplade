#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=save_embs_canard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hai.le@etu.sorbonne-universite.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=64000
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil"
python save_ctx_embs.py
