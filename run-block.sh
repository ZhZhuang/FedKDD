#!/bin/sh
# cd py脚本的路径；

# 结合 扩散模型的知识蒸馏方法，我们暂时就叫他 FedPro
# 公共数据集 5000
# 扩充数据集和原先的数据集同样的数量，在cifar10中生成5w， 在fmnist中生成6w


#########################################


echo "基于知识蒸馏的联邦学习"
echo "KTpFL_pro 同构构型"



############   第一类
python main_KTpFL_pro.py --name FedPro_20b --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --gen_path img_cifar10_100t_alpha1.0_5w  --n_block 20

python main_KTpFL_pro.py --name FedPro_40b --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --gen_path img_cifar10_100t_noniid4_5w --n_block 40

python main_KTpFL_pro.py --name FedPro_100b --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --n_cls 4 \
                      --gen_path img_cifar10_100t_noniid4_5w --n_block 100


#####################  第二类

python main_KTpFL_pro.py --name FedPro_20b --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --gen_path img_cifar10_100t_alpha1.0_5w  --n_block 20

python main_KTpFL_pro.py --name FedPro_40b --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --gen_path img_cifar10_100t_alpha1.0_5w  --n_block 40

python main_KTpFL_pro.py --name FedPro_100b --n_clients 10 --dataset Cifar10  \
                      --round 50 --local_epoch 1  --public_training_round 1 \
                      --dirichlet --alpha 1.0 \
                      --gen_path img_cifar10_100t_alpha1.0_5w  --n_block 100


echo "结束训练"

echo "即将关机..."
#sudo shutdown -h now
