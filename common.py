import copy
import yaml
import os
import warnings
import random
import numpy as np
import torch
from torch.backends import cudnn
import argparse
import json
import logging
from torchvision import transforms
from datasets.BasicDataset import BasicDataset
from datasets.augmentation.randaugment import RandAugment
from models.ResNet import ResNet18
from utils import get_public_data, cifar_noniid, get_user_data, \
    fmnist_noniid, cifar_noniid_dirichlet, fmnist_noniid_dirichlet, cifar_noniid_byclass, fmnist_noniid_byclass
from models import client,client_fedprox,client_fednova,client_scaffold

def dict2namespace( config ):
    namespace = argparse.Namespace()    # ？
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_models( config, logger ):
    n_clients= config.n_clients

    # 准备数据
    path = config.dataset_path
    if config.noniid:
        # non iid   Dirichlet分布
        if config.dataset == "Cifar10":
            if config.dirichlet:
                # dirichlet 分布
                train_data_dict_list, test_data_dict_list = cifar_noniid_dirichlet(n_clients,
                                                                                   alpha= config.alpha,
                                                                                   path= path)
            else:
                # train_data_dict_list, test_data_dict_list = cifar_noniid(n_clients, path=path)
                train_data_dict_list, test_data_dict_list = cifar_noniid_byclass( config.n_cls )    # 10 用户默认
        elif config.dataset == "FashionMNIST":
            if config.dirichlet:
                train_data_dict_list, test_data_dict_list = fmnist_noniid_dirichlet(n_clients, alpha=config.alpha, path= path)
            else:
                # train_data_dict_list, test_data_dict_list = fmnist_noniid(num_users= n_clients, path= path)
                train_data_dict_list, test_data_dict_list = fmnist_noniid_byclass( config.n_cls )
    else:
        # iid
        if config.dataset == "Cifar10":
            train_data_dict_list = get_user_data(n_clients, train=True, dataname="Cifar10")
            test_data_dict_list = get_user_data(n_clients, train=False, dataname="Cifar10")
        elif config.dataset == "FashionMNIST":
            train_data_dict_list = get_user_data(n_clients, train=True, dataname="FashionMNIST")
            test_data_dict_list = get_user_data(n_clients, train=True, dataname="FashionMNIST")

    is_gray = False
    if config.dataset == "FashionMNIST":
        is_gray = True

    # 初始化用户
    client_list = []

    for i in range(n_clients):

        in_ch = 3
        if config.dataset != "Cifar10":
            in_ch = 1
        C_model = ResNet18(in_ch= in_ch)
        data = train_data_dict_list[i]["sub_data"]
        targets = train_data_dict_list[i]["sub_targets"]
        test_data = test_data_dict_list[i]["sub_data"]
        test_targets = test_data_dict_list[i]["sub_targets"]

        ################# 获取每类样本的数量，并保存在文件中
        count = [0 for _ in range(config.classes)]
        for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
            count[c] += 1
        out = {"distribution": count }
        output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
        # if not os.path.exists(output_file):
        #     os.makedirs(output_file, exist_ok=True)
        with open(output_file, 'w') as w:
            json.dump(out, w)
        #################

        if config.dataset == "Cifar10":
            transform = transforms.Compose([
                                            transforms.Pad(4),
                                            transforms.RandomHorizontalFlip(),  # ? 水平翻转
                                            transforms.RandomCrop(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ])
            test_transform = transforms.Compose([transforms.Resize(32),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                 ])
        elif config.dataset == "FashionMNIST":
            transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),  # ? 水平翻转
                transforms.RandomCrop(28),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # error 变为 3 通道
                transforms.Normalize( 0.5, 0.5)
            ])
            test_transform = transforms.Compose([transforms.Resize(28),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(0.5, 0.5)
                                                 ])
        dataset = BasicDataset(data, targets, transform=transform,onehot= False)

        # =================数据增强 ！！！===============
        if config.use_aug:
            data_2 = copy.deepcopy(data)
            targets_2 = copy.deepcopy(targets)
            strong_transform = copy.deepcopy(transform)
            strong_transform.transforms.insert(0, RandAugment(3, 5, is_gray= is_gray))  # 进行三种数据增强
            dataset_2 = BasicDataset(data_2, targets_2, transform=strong_transform, onehot= False)
            dataset += dataset_2


        test_dataset = BasicDataset(test_data, test_targets, transform=test_transform,onehot= False)
        dataloader = torch.utils.data.DataLoader(dataset, config.batch_size,
                                                 shuffle=True,
                                                 num_workers= config.num_works)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 256,
                                                 shuffle=False,
                                                 num_workers= config.num_works)

        condidate = {
            "FedAvg": client.Client,
            "FedAvg+": client.Client,
            "FedNova": client_fednova.Client,
            "FedProx": client_fedprox.Client,
            "Scaffold": client_scaffold.Client,
            "KT-pFL": client.Client
        }

        # client = Client(config,
        #                 C_model=C_model,
        #                 client_idx=i,
        #                 dataloader= dataloader,
        #                 dataset = dataset,
        #                 t_dataloader= test_dataloader,
        #                 logger= logger
        #                 )

        client_ = condidate[config.name](config,
                        C_model=C_model,
                        client_idx=i,
                        dataloader=dataloader,
                        dataset=dataset,
                        t_dataloader=test_dataloader,
                        logger=logger
                        )

        # Optimizers
        # optimizer_C = torch.optim.SGD(C_model.parameters(),lr=0.001) 0.01
        optimizer_C = torch.optim.Adam(C_model.parameters(), lr = config.lr)  #更适合resnet18
        client_.set_optimizer( optimizer_C )
        client_list.append( client_ )

    return client_list , transform

def str2bool(v):        # 吸收
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init():

    # 添加参数
    parser = argparse.ArgumentParser(description=globals()["__doc__"])  # ？
    parser.add_argument("--name", type=str, default="FedAvg", help="FedAvg/FedProx/FedNova/Scaffold")
    # parser.add_argument("--config", type=str, default="fedavg.yaml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="save", help="Path to save results")
    parser.add_argument("--n_clients", type=int, default=10, help="number of clients，5，10，20")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path of dataset")
    parser.add_argument("--dataset", type=str, default="Cifar10", help="Cifar10/FashionMNIST")
    parser.add_argument("--num_works", type=int, default=10, help="Number of classes")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size 128 64")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--round", type=int, default=100, help="communication round")
    parser.add_argument("--local_epoch", type=int, default=1, help="local epoch")
    # 算法的特殊参数
    parser.add_argument("--mu", type=float, default= 0.01 , help="Parameters of FedPorx")
    parser.add_argument("--rho", type=float, default=0.0, help="Parameters of FedNova")
    # 数据增强
    parser.add_argument("--use_aug", type=str2bool,default=False, help="user dada augment or not")
    # non-IID 参数
    parser.add_argument("--noniid", type=str2bool, default=True, help="noniid or not")  # 把 True 固定了 ，type的问题，解决，添加str2bool
    # parser.add_argument("--dirichlet", type=bool, default=False, help="dirichlet or not") # 有问题
    parser.add_argument("--dirichlet",action="store_true", help="dirichlet or not")
    parser.add_argument("--alpha", type=float, default=1.0, help="control dirichlet")
    parser.add_argument("--n_cls", type=int, default=4, help="n classes per client have: 2 | 4")

    # args = parser.parse_args()

    # # 读取配置文件
    # with open(os.path.join("configs", args.config), "r") as f:
    #     config = yaml.safe_load(f)
    # new_config = dict2namespace(config)     #

    new_config = parser.parse_args()

    # 保存路径
    save_dir = None
    if new_config.noniid == False:
        save_dir = os.path.join(new_config.save_dir,
                                new_config.name + f"_{new_config.dataset}" + \
                                f"_{new_config.n_clients}" + f"_iid" )
    elif new_config.noniid and new_config.dirichlet == False:
        save_dir = os.path.join(new_config.save_dir,
                                new_config.name + f"_{new_config.dataset}" + \
                                f"_{new_config.n_clients}c" + f"_noniid" + \
                                f"_{new_config.n_cls}")
    elif new_config.noniid and new_config.dirichlet:
        save_dir = os.path.join(new_config.save_dir,
                                new_config.name  + f"_{new_config.dataset}" + \
                                f"_{new_config.n_clients}c" + \
                                f"_dirichlet_alpha_{new_config.alpha}")
    new_config.save_dir = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)    # True：创建目录的时候，如果已存在不报错。

    # 获取设备信息
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # 将配置信息保存
    with open(os.path.join(new_config.save_dir, "config.yml"), "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

    seed = new_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True  # 随机数种子seed确定时，模型的训练结果将始终保持一致
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    return new_config
