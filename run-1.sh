#!/bin/sh
# cd py脚本的路径；

echo "开始测试......"
echo "Cifar10 数据集上，第一种 noniid 分布, 每个用户最多 4 类数据......"

echo "FedNova"
python main_fednova.py --name FedNova --n_clients 5 --dataset Cifar10 --local_epoch 5
wait
python main_fednova.py --name FedNova --n_clients 10 --dataset Cifar10 --local_epoch 5
wait
python main_fednova.py --name FedNova --n_clients 20 --dataset Cifar10 --local_epoch 5
wait


echo "开始测试......"
echo "FashionMNIST 数据集上，第一种 noniid 分布, 每个用户最多 4 类数据......"
echo "FedNova"
python main_fednova.py --name FedNova --n_clients 5 --dataset FashionMNIST --local_epoch 5
wait
python main_fednova.py --name FedNova --n_clients 10 --dataset FashionMNIST --local_epoch 5
wait
python main_fednova.py --name FedNova --n_clients 20 --dataset FashionMNIST --local_epoch 5
wait
echo "Scaffold"
python main_scaffold.py --name Scaffold --n_clients 5 --dataset FashionMNIST --local_epoch 5
wait
python main_scaffold.py --name Scaffold --n_clients 10 --dataset FashionMNIST --local_epoch 5
wait
python main_scaffold.py --name Scaffold --n_clients 20 --dataset FashionMNIST --local_epoch 5

echo "第一种 noniid 分布 结束测试......"

echo "Dirichlet 分布 开始测试......"
echo "FashionMNIST 数据集"
echo "FedAvg"
python main_fedavg.py --name FedAvg --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_fedavg.py --name FedAvg --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait

echo "FedProx"
python main_fedprox.py --name FedProx --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_fedprox.py --name FedProx --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait

echo "FedNova"
python main_fednova.py --name FedNova --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_fednova.py --name FedNova --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait
echo "Scaffold"
python main_scaffold.py --name Scaffold --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_scaffold.py --name Scaffold --dataset FashionMNIST --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait

echo "Cifar10 数据集"
echo "FedAvg"
python main_fedavg.py --name FedAvg --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_fedavg.py --name FedAvg --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait

echo "FedProx"
python main_fedprox.py --name FedProx --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_fedprox.py --name FedProx --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait

echo "FedNova"
python main_fednova.py --name FedNova --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_fednova.py --name FedNova --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait
echo "Scaffold"
python main_scaffold.py --name Scaffold --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 0.1
wait
python main_scaffold.py --name Scaffold --dataset Cifar10 --n_clients 10 --round 100 --local_epoch 5 \
                      --dirichlet --alpha 1.0
wait