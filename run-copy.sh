#!/bin/sh
# cd py脚本的路径；

echo "开始测试......"
echo "Cifar10 数据集上，第一种 noniid 分布, 每个用户最多 4 类数据......"
#echo "FedAvg"
#python main_fedavg.py --name FedAvg --n_clients 5 --dataset Cifar10 --local_epoch 5
#wait
#python main_fedavg.py --name FedAvg --n_clients 10 --dataset Cifar10 --local_epoch 5
#wait
#python main_fedavg.py --name FedAvg --n_clients 20 --dataset Cifar10 --local_epoch 5
#wait
#echo "FedProx"
#python main_fedprox.py --name FedProx --n_clients 5 --dataset Cifar10 --local_epoch 5
#wait
#python main_fedprox.py --name FedProx --n_clients 10 --dataset Cifar10 --local_epoch 5
#wait
#python main_fedprox.py --name FedProx --n_clients 20 --dataset Cifar10 --local_epoch 5
#wait
echo "FedNova"
python main_fednova.py --name FedNova --n_clients 5 --dataset Cifar10 --local_epoch 5
wait
python main_fednova.py --name FedNova --n_clients 10 --dataset Cifar10 --local_epoch 5
wait
python main_fednova.py --name FedNova --n_clients 20 --dataset Cifar10 --local_epoch 5
wait
#echo "Scaffold"
#python main_scaffold.py --name Scaffold --n_clients 5 --dataset Cifar10 --local_epoch 5
#wait
#python main_scaffold.py --name Scaffold --n_clients 10 --dataset Cifar10 --local_epoch 5
#wait
#python main_scaffold.py --name Scaffold --n_clients 20 --dataset Cifar10 --local_epoch 5
#wait

echo "开始测试......"
echo "FashionMNIST 数据集上，第一种 noniid 分布, 每个用户最多 4 类数据......"
#echo "FedAvg"
#python main_fedavg.py --name FedAvg --n_clients 5 --dataset FashionMNIST --local_epoch 5
#wait
#python main_fedavg.py --name FedAvg --n_clients 10 --dataset FashionMNIST --local_epoch 5
#wait
#python main_fedavg.py --name FedAvg --n_clients 20 --dataset FashionMNIST --local_epoch 5
#wait
#echo "FedProx"
#python main_fedprox.py --name FedProx --n_clients 5 --dataset FashionMNIST --local_epoch 5
#wait
#python main_fedprox.py --name FedProx --n_clients 10 --dataset FashionMNIST --local_epoch 5
#wait
#python main_fedprox.py --name FedProx --n_clients 20 --dataset FashionMNIST --local_epoch 5
#wait
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

echo "结束测试......"

#wait能等待前一个脚本执行完毕，再执行下一个条命令；