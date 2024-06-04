#!/bin/sh
# cd py脚本的路径；


##  测试代码是否能用


echo "基于知识蒸馏的联邦学习"
echo "FedMD"

#######################  同构网络  ###########################
#python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST  \
#                      --round 50 --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0
#
#python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST  \
#                      --round 50 --local_epoch 1  --public_training_round 1 \
#                      --n_cls 4
#
#python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
#                      --round 50 --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0
#
#python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
#                      --round 50 --local_epoch 1  --public_training_round 1 \
#                      --n_cls 4


#######################  异构网络  ######################

python main_FedMD.py --name FedMD_H --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --heterogeneity

python main_FedMD.py --name FedMD_H --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --heterogeneity

python main_FedMD.py --name FedMD_H --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --heterogeneity

python main_FedMD.py --name FedMD_H --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --heterogeneity


echo "即将关机..."
sudo shutdown -h now