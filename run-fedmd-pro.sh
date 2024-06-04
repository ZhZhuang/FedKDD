#!/bin/sh
# cd py脚本的路径；

##  测试代码是否能用


# 结合 扩散模型的知识蒸馏方法，我们暂时就叫他 FedPro
# 公共数据集 5000
# 扩充数据集和原先的数据集同样的数量，在cifar10中生成5w， 在fmnist中生成6w


#########################################


echo "基于知识蒸馏的联邦学习"
echo "FedMD-pro 同构构型"

python main_FedMD_pro.py --name FedMD_pro --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --gen_path img_fmnist_100t_alpha1.0_6w

python main_FedMD_pro.py --name FedMD_pro --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --gen_path img_fmnist_100t_noniid4_6w

python main_FedMD_pro.py --name FedMD_pro --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --gen_path img_cifar10_100t_alpha1.0_5w

python main_FedMD_pro.py --name FedMD_pro --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --gen_path img_cifar10_100t_noniid4_5w



##########################      下面的实验感觉没必要做了      #########################
echo "基于知识蒸馏的联邦学习"
echo "KTpFL 异构网络"

python main_FedMD_pro.py --name FedMD_pro_H --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --heterogeneity \
                      --gen_path img_fmnist_100t_alpha1.0_6w


python main_FedMD_pro.py --name FedMD_pro_H --n_clients 10 --dataset FashionMNIST  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --heterogeneity \
                      --gen_path img_fmnist_100t_noniid4_6w

python main_FedMD_pro.py --name FedMD_pro_H --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --heterogeneity \
                      --gen_path img_cifar10_100t_alpha1.0_5w

python main_FedMD_pro.py --name FedMD_pro_H --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --heterogeneity \
                      --gen_path img_cifar10_100t_noniid4_5w


echo "结束训练"

echo "即将关机..."
#sudo shutdown -h now
