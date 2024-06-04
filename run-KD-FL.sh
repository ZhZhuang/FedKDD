#!/bin/sh
# cd py脚本的路径；

echo "基于知识蒸馏的联邦学习"

##  ################
python main_KTpFL.py --name KTpFL --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 0.1

python main_KTpFL.py --name KTpFL --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 1.0

python main_KTpFL.py --name KTpFL --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 2

python main_KTpFL.py --name KTpFL --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 4

##  ################

##  ################
python main_KTpFL.py --name KTpFL --n_clients 10 --dataset FashionMNIST  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 0.1

python main_KTpFL.py --name KTpFL --n_clients 10 --dataset FashionMNIST  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 1.0

python main_KTpFL.py --name KTpFL --n_clients 10 --dataset FashionMNIST \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 2

python main_KTpFL.py --name KTpFL --n_clients 10 --dataset FashionMNIST \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 4
##  ################

####  FedMD
##  ################
python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 0.1

python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 1.0

python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 2

python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 4
#
##  ################

##  ################
python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 0.1

python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST  \
                      --local_epoch 2  --public_training_round 2 \
                      --dirichlet --alpha 1.0

python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 2

python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST \
                      --local_epoch 2  --public_training_round 2 \
                       --n_cls 4

##  ################

echo "结束测试......"
