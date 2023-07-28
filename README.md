# Hamiltonian-GNN
Node Embedding from Neural Hamiltonian Orbits in Graph Neural Networks

This repository contains the code for our ICML 2023 accepted paper, *Node Embedding from Neural Hamiltonian Orbits in Graph Neural Networks*.
## Table of Contents

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Hyperparameters](#Hyperparameters)
- [Reference](#reference)
- [Citation](#citation)

## Requirements

To install the required dependencies, refer to the environment.yaml file

## Hyperparameters

the function H_net in the paper saved in the ./layers folder 
- H_1.py refers to Equation(20)   --odemap h1extend
- H_2.py refers to Equation(21)	  --odemap h2extend
- H_3.py refers to Equation(26)	  --odemap h3extend
- H_4.py refers to Equation(22)	  --odemap h4extend
- H_6.py refers to Equation(25)	  --odemap h6extend
- H_8.py refers to Equation(23)	  --odemap h8extend

```bash
Equation(20) 
python main_nc.py --odemap h1extend --dataset cora --num_layers 2 --hidden 64 --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.5 --act None
python main_nc.py --odemap h1extend --dataset citeseer --num_layers 2 --hidden 64 --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.2 --act None
python main_nc.py --odemap h1extend --dataset pubmed --num_layers 2 --hidden 128 --lr 0.01 --decay 0.001 --dropout 0 --step_size 1.0 --act relu
python main_nc.py --odemap h1learn --dataset airport --num_layers 2 --hidden 128 --lr 0.001 --decay 0.0001 --dropout 0 --step_size 1.0 --act relu
python main_nc.py --odemap h1extend --dataset disease_nc --num_layers 2 --hidden 128 --lr 0.01 --decay 0.0001 --dropout 0.1 --step_size 1.0 --act None --patience 500
Equation(21)	 
python main_nc.py --odemap h2extend --dataset cora --num_layers 5  --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.5 --act relu --hidden 128
python main_nc.py --odemap h2extend --dataset citeseer --num_layers 3  --lr 0.001 --decay 0.01 --dropout 0 --step_size 1.0 --act None --hidden 128
python main_nc.py --odemap h2extend --dataset pubmed --num_layers 2  --lr 0.001 --decay 0.001 --dropout 0 --step_size 1.0 --act relu --hidden 128
python main_nc.py --odemap h2learn --dataset airport --num_layers 2  --lr 0.01 --decay 0.0001 --dropout 0 --step_size 1.0 --act None --hidden 128 --patience 500
python main_nc.py --odemap h2extend --dataset disease_nc --num_layers 3  --lr 0.01 --decay 0.0001 --dropout 0.2 --step_size 1.0 --act None --hidden 128 --patience 500
Equation(26)
python main_nc.py --odemap h3extend --dataset cora --num_layers 3 --lr 0.001 --decay 0.01 --dropout 0 --step_size 0.2 --act None --hidden 128 --kdim 6
python main_nc.py --odemap h3extend --dataset citeseer --num_layers 3 --lr 0.001 --decay 0.01 --dropout 0 --step_size 0.2 --act None --hidden 128 
python main_nc.py --odemap h3extend --dataset pubmed --num_layers 2 --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.5 --act relu --hidden 128 
python main_nc.py --odemap h3learn --dataset airport --num_layers 4  --lr 0.001 --decay 0.0001 --dropout 0 --step_size 1.0 --act None --hidden 128 
python main_nc.py --odemap h3extend --dataset disease_nc --num_layers 2  --lr 0.001 --decay 0.001 --dropout 0 --step_size 1.0 --act None --hidden 128 --patience 500

Equation(22)
python main_nc.py --odemap h4extend --dataset cora --num_layers 2  --lr 0.01 --decay 0.001 --dropout 0 --step_size 1.0 --act relu --hidden 64
python main_nc.py --odemap h4extend --dataset citeseer --num_layers 2  --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.2 --act None --hidden 128
python main_nc.py --odemap h4extend --dataset pubmed --num_layers 2  --lr 0.001 --decay 0.001 --dropout 0 --step_size 0.5 --act None --hidden 128
python main_nc.py --odemap h4learn --dataset airport --num_layers 3  --lr 0.001 --decay 0.0001 --dropout 0 --step_size 0.5 --act relu --hidden 128
python main_nc.py --odemap h4extend --dataset disease_nc --num_layers 2  --lr 0.01 --decay 0.0001 --dropout 0 --step_size 0.2 --act None --hidden 64 --patience 500

Equation(25)
python main_nc.py --odemap h6extend --dataset cora --num_layers 2 --hidden 64 --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.5 --act relu
python main_nc.py --odemap h6extend --dataset citeseer --num_layers 2 --hidden 64 --lr 0.001 --decay 0.01 --dropout 0 --step_size 0.2 --act None
python main_nc.py --odemap h6extend --dataset pubmed --num_layers 2 --hidden 16 --lr 0.001 --decay 0.01 --dropout 0 --step_size 1.0 --act None
python main_nc.py --odemap h6learn --dataset airport --num_layers 2 --hidden 128 --lr 0.001 --decay 0.0001 --dropout 0 --step_size 1.0 --act relu
python main_nc.py --odemap h6extend --dataset disease_nc --num_layers 2 --hidden 128 --lr 0.01 --decay 0.0001 --dropout 0 --step_size 0.5 --act relu --patience 500


Equation(23)
python main_nc.py --odemap h8extend --dataset cora --num_layers 2 --hidden 32 --lr 0.01 --decay 0.001 --dropout 0 --step_size 0.2 --act relu --cuda 0 --patience 100 --epoch 2000 --vt fc --odemethod euler --seed 1234

python main_nc.py --odemap h8extend --dataset citeseer --num_layers 2 --hidden 64 --lr 0.001 --decay 0.01 --dropout 0 --step_size 1.0 --act None

python main_nc.py --odemap h8extend --dataset cora --num_layers 2 --hidden 32 --lr 0.001 --decay 0.001 --dropout 0 --step_size 0.2 --act relu

python main_nc.py --odemap h8learn --dataset airport --num_layers 1 --hidden 64 --lr 0.01 --decay 0.0001 --dropout 0 --step_size 0.5 --act relu

python main_nc.py --odemap h8extend --dataset disease_nc --num_layers 2 --hidden 64 --lr 0.001 --decay 0.0001 --dropout 0 --step_size 0.2 --act None --patience 500



```


## Citation

If you find our helpful, consider to cite us:
```bash
@INPROCEEDINGS{KanZhaSon:C23,
author = {Qiyu Kang and Kai Zhao and Yang Song and Sijie Wang and Wee Peng Tay},
title = {Node Embedding from Neural {Hamiltonian} Orbits in Graph Neural Networks},
booktitle = {Proc. International Conference on Machine Learning},
volume = {},
pages = {},
month = {Jul.},
year = {2023},
address = {Haiwaii, USA},
}

