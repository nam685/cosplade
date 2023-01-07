#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=orqa_splade_index
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hai.le@etu.sorbonne-universite.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=64000
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil"
cd /data/lenam/splade
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/orqa/orqa_splade_index/0/index" config.out_dir="/data/lenam/corpus/orqa/orqa_splade_index/0/outputs" data.COLLECTION_PATH="/data/lenam/corpus/orqa/raw/orqa_0"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/orqa/orqa_splade_index/2048000/index" config.out_dir="/data/lenam/corpus/orqa/orqa_splade_index/2048000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/orqa/raw/orqa_2048000"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/orqa/orqa_splade_index/4096000/index" config.out_dir="/data/lenam/corpus/orqa/orqa_splade_index/4096000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/orqa/raw/orqa_4096000"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/orqa/orqa_splade_index/6144000/index" config.out_dir="/data/lenam/corpus/orqa/orqa_splade_index/6144000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/orqa/raw/orqa_6144000"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/orqa/orqa_splade_index/8192000/index" config.out_dir="/data/lenam/corpus/orqa/orqa_splade_index/8192000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/orqa/raw/orqa_8192000"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/orqa/orqa_splade_index/10240000/index" config.out_dir="/data/lenam/corpus/orqa/orqa_splade_index/10240000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/orqa/raw/orqa_10240000"







