# High-order Pair-reduced Neural Network Architecture for Global Potential Energy Surface Exploration Across the Periodic Table

## Description
This repository implements the High-order Pair-reduced Neural Network (HPNN), a machine learning model designed for efficient and accurate atomic simulations. HPNN employs a hierarchical angular interaction scheme with reduced pair dimensions, incorporating spherical harmonics up to l=6 for high-fidelity predictions of atomic energies and forces.



## Install
### miniconda3
```
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh

./miniconda3/bin/conda init bash
```

### python
```
conda create --name hpnn python=3.11
# 激活conda hpnn
conda activate hpnn
```
### python packages
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install e3nn
```
## Code Structure
```
src:[./src]
|--__init__.py
|--data
|      |--train                                         # 
|      |      |--raw
|      |      |      |--force.arc                       #
|      |      |      |--structure.arc                   # 
|--config.yml                                           # 
|--run_amp.py                                           # single card main
|--run_ddp.py                                           # ddp main
|--model.py                                             # model
|--model_block.py                                       # model
|--model_pth
|      |--256_3_128_train_1E5F1S.pth                    # train best model
|      |--256_3_128_valid_1E5F1S.pth                    # valid best model
|--logs
|      |--XXXXXX_512_3_128_20240301.log                 # log
|      |--debug                                         # gpu detail
|--utils
|      |--__init__.py
|      |--common.py
|      |--run_common.py
|      |--load_common.py
|      |--load_valid.py                                 
|      |--load_arc.py
|      |--load_vasp.py
|      |--valid.py                                      #
|--load_data.py                                         # load data to pyg dataset
|--readme.md
```
## How to Run
```
python load_data.py
# single gpu
python run_amp.py
```
