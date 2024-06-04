#!/bin/sh
# cd py脚本的路径；

echo "开始测试......"
echo "Cifar10 数据集上，第一种 noniid 分布, 每个用户最多 2 类数据......"

echo "FedAvg"
# 测试极限
python main_fedavg.py --name FedAvg --n_clients 10 --dataset Cifar10 --local_epoch 1 --n_cls 4 --round 100 --num_works 8
#top acc : 0.7828, at 184 round!

#python main_fedavg.py --name FedAvg --n_clients 10 --dataset FashionMNIST --local_epoch 5 --n_cls 2
#wait
#
#echo "FedProx"
#python main_fedprox.py --name FedProx --n_clients 10 --dataset Cifar10  --local_epoch 5 --n_cls 2
#python main_fedprox.py --name FedProx --n_clients 10 --dataset FashionMNIST --local_epoch 5 --n_cls 2
#wait
#
#echo "FedNova"
#python main_fednova.py --name FedNova --n_clients 10 --dataset Cifar10 --local_epoch 5 --n_cls 2
#python main_fednova.py --name FedNova --n_clients 10 --dataset FashionMNIST --local_epoch 5 --n_cls 2
#wait
#
#echo "Scaffold"
#python main_scaffold.py --name Scaffold --n_clients 10 --dataset Cifar10 --local_epoch 5 --n_cls 2
#python main_scaffold.py --name Scaffold --n_clients 10 --dataset FashionMNIST --local_epoch 5 --n_cls 2
#wait


echo "结束测试......"
