#!/bin/sh
# cd py脚本的路径；


##  测试代码是否能用


echo "基于知识蒸馏的联邦学习"

python main_FedMD_pro.py --name FedMD_pro --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --gen_path img_fmnist_100t_alpha1.0_5k
#                      --gen_path img_fmnist_100t_alpha1.0_6w

#python main_KTpFL_pro.py --name FedPro --n_clients 10 --dataset Cifar10  \
#                      --round 50 --local_epoch 1  --public_training_round 1 \
#                      --n_cls 4 \
#                      --gen_path img_cifar10_100t_noniid4_5w

#python main_KTpFL_pro.py --name FedPro --n_clients 10 --dataset FashionMNIST  \
#                      --round 2 --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0 \
#                      --gen_path img_fmnist_100t_alpha1.0_5k  \
#                      --public_path img_fmnist_100t_alpha1.0_5k



#python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST  \
#                      --round 2 --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0

#python main_FedMD.py --name FedMD_H --n_clients 10 --dataset FashionMNIST  \
#                      --round 2 --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0 \
#                      --heterogeneity

#python main_FedMD.py --name FedMD_H --n_clients 10 --dataset Cifar10  \
#                      --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0 \
#                      --heterogeneity



#python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
#                      --local_epoch 1  --public_training_round 1 \
#                      --dirichlet --alpha 1.0


#python main_FedMD.py --name FedMD --n_clients 10 --dataset Cifar10  \
#                      --local_epoch 1  --public_training_round 1 \
#                      --n_cls 4run

wait

#python main_FedMD.py --name FedMD --n_clients 10 --dataset FashionMNIST  \
#                      --local_epoch 1  --public_training_round 1 \
#                      --n_cls 4



# 很有问题，半小时一个round，其次，现存上不去，GPU利用率也上不去

#python main_KTpFL.py --name KTpFL --n_clients 10 --dataset Cifar10  \
#                      --local_epoch 2  --public_training_round 1 \
#                      --dirichlet --alpha 0.1

echo "结束测试......"
echo "结束测试......"
