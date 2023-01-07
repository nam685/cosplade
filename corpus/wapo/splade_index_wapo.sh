#!/bin/bash

#SBATCH --job-name=wapo_splade_index
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hai.le@etu.sorbonne-universite.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd /data/lenam/splade
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/wapo/wapo_splade_index/0/index" config.out_dir="/data/lenam/corpus/wapo/wapo_splade_index/0/outputs" data.COLLECTION_PATH="/data/lenam/corpus/wapo/raw/wapo_0"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/wapo/wapo_splade_index/2048000/index" config.out_dir="/data/lenam/corpus/wapo/wapo_splade_index/2048000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/wapo/raw/wapo_2048000"
