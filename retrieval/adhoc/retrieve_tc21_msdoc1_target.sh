#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=retrieve_tc21_msdoc1_target
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hai.le@etu.sorbonne-universite.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=48000
#SBATCH --mincpus=4

cd /data/lenam/splade
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil"
python -m src.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir=/data/lenam/corpus/msdoc1/msdoc1_splade_index config.out_dir=/data/lenam/retrieval/adhoc/tc21_msdoc1_target.out data.Q_COLLECTION_PATH=/data/lenam/topics/tsv/tc21_target.tsv data.EVAL_QREL_PATH=/data/lenam/qrel/qr21.json

