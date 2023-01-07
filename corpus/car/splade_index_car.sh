#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=car_splade_index
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hai.le@etu.sorbonne-universite.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=64000

cd /data/lenam/splade
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/0/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/0/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_0"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_2048000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_2048000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_2048000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_4096000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_4096000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_4096000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_6144000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_6144000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_6144000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_8192000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_8192000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_8192000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_10240000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_10240000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_10240000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_12288000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_12288000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_12288000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_14336000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_14336000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_14336000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_16384000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_16384000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_16384000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_18432000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_18432000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_18432000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_20480000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_20480000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_20480000"
python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_22528000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_22528000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_22528000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_24576000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_24576000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_24576000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_26624000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_26624000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_26624000"
#python3 -m src.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir="/data/lenam/corpus/car/car_splade_index/car_28672000/index" config.out_dir="/data/lenam/corpus/car/car_splade_index/car_28672000/outputs" data.COLLECTION_PATH="/data/lenam/corpus/car/raw/car_28672000"

