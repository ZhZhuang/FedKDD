#!/bin/sh
# cd py脚本的路径；

# FedAvg+ 的对比实验

echo "开始测试......"
echo "Cifar10 数据集上，第一种 noniid 分布, 每个用户最多 2 类数据......"

# 默认 round：100 --num_works：10
echo "FedAvg+"
#python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset Cifar10 --local_epoch 2 --n_cls 2 \
#                      --use_aug True
#
#python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset Cifar10 --local_epoch 2 --n_cls 4 \
#                      --use_aug True
#
#python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset FashionMNIST --local_epoch 2 --n_cls 2 \
#                      --use_aug True
#
#python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset FashionMNIST --local_epoch 2 --n_cls 4 \
#                      --use_aug True
wait

python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset Cifar10 --local_epoch 2  \
                      --dirichlet --alpha 0.1 \
                      --use_aug True

python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset Cifar10 --local_epoch 2 \
                      --dirichlet --alpha 1.0 \
                      --use_aug True

python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset FashionMNIST --local_epoch 2 \
                      --dirichlet --alpha 0.1 \
                      --use_aug True

python main_fedavg.py --name FedAvg+ --n_clients 10 --dataset FashionMNIST --local_epoch 2 \
                      --dirichlet --alpha 1.0 \
                      --use_aug True

echo "结束测试......"
