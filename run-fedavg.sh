#!/bin/sh
# cd py脚本的路径；

echo "开始测试......"
echo "Cifar10 数据集上，第一种 noniid 分布, 每个用户最多 2 类数据......"

echo "FedAvg"
# 测试极限

python main_fedavg.py --name FedAvg --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  \
                      --dirichlet --alpha 1.0

python main_fedavg.py --name FedAvg --n_clients 10 --dataset FashionMNIST \
                      --round 50 --local_epoch 1 --n_cls 4


#python main_fedavg.py --name FedAvg --n_clients 10 --dataset Cifar10 \
#                       --round 50 --local_epoch 1  \
#                       --dirichlet --alpha 1.0
#
#python main_fedavg.py --name FedAvg --n_clients 10 --dataset Cifar10 \
#                      --round 50  --local_epoch 1 --n_cls 4
#

echo "结束测试......"
