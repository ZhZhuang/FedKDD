import argparse
import copy
import json
import logging
import os
import random

import torch
import yaml
# from imgaug.augmenters import RandAugment
from torch.backends import cudnn
from torchvision import transforms
from wandb.util import np

from core_fedmd_pro import fedmd_train_pro
from datasets.BasicDataset import BasicDataset
from datasets.fileDataset import fileDataset
from models import client
from models.LeNet import LeNet
from models.ResNet import ResNet18
from utils import get_logger, cifar_noniid_dirichlet, cifar_noniid_byclass, fmnist_noniid_dirichlet, \
    fmnist_noniid_byclass, get_user_data, get_public_data, get_data_random_idx

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

        if config.heterogeneity:
            print("异构的神经网络")
            if i<5:
                # 前5个是LeNet
                C_model = LeNet(in_ch)
            else:
                C_model = ResNet18(in_ch= in_ch)
        else:
            C_model = ResNet18(in_ch= in_ch)
        data = train_data_dict_list[i]["sub_data"]
        targets = train_data_dict_list[i]["sub_targets"]
        test_data = test_data_dict_list[i]["sub_data"]
        test_targets = test_data_dict_list[i]["sub_targets"]

        ################# 获取每类样本的数量，并保存在文件中
        count = [0 for _ in range(config.classes)]
        for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
            count[c] += 1
        # out = {"distribution": count }


        test_count = [0 for _ in range(config.classes)]
        for c in test_targets:  # lb_targets 为 0 ～ 9 ， 有操作
            test_count[c] += 1
        out = {"distribution": count,
            "test_distribution": test_count}
        # if not os.path.exists(output_file):
        #     os.makedirs(output_file, exist_ok=True)

        output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
        with open(output_file, 'w') as w:
            json.dump(out, w)       # 训练数据分布
        #################



        if config.dataset == "Cifar10":
            transform = transforms.Compose([
                                            transforms.Pad(4),
                                            transforms.RandomHorizontalFlip(),  # ? 水平翻转
                                            transforms.RandomCrop(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ]
                                           )
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
                 ]
            )
            test_transform = transforms.Compose([transforms.Resize(28),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(0.5, 0.5)
                                                 ])
        dataset = BasicDataset(data, targets, transform=transform,onehot= False)

        # =================数据增强 ！！！===============
        # if config.use_aug:
        #     data_2 = copy.deepcopy(data)
        #     targets_2 = copy.deepcopy(targets)
        #     strong_transform = copy.deepcopy(transform)
        #     strong_transform.transforms.insert(0, RandAugment(3, 5, is_gray= is_gray))  # 进行三种数据增强
        #     dataset_2 = BasicDataset(data_2, targets_2, transform=strong_transform, onehot= False)
        #     dataset += dataset_2


        test_dataset = BasicDataset(test_data, test_targets, transform=test_transform,onehot= False)
        dataloader = torch.utils.data.DataLoader(dataset, config.batch_size,
                                                 shuffle=True,
                                                 num_workers= config.num_works)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 256,
                                                 shuffle=False,
                                                 num_workers= config.num_works)

        client_ = client.Client(config,
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

    # ==================== 生成数据集的处理 =====================
    path_data_file = os.path.join("data_gen", config.gen_path)
    # path_data_file = config.gen_path
    # cifar10 扩散模型生成样本，400 epoch，5 用户，10w， 联邦训练的得到
    g_dataset = fileDataset(path_data_file,
                            transform=transform,
                            is_gray=is_gray,
                            )

    dict_users = get_data_random_idx(g_dataset, config.n_block)
    # dict_users = get_data_random_idx(g_dataset, config.n_clients)

    data_list = []
    for i in range(n_clients):
        idxs = dict_users[i]  # 数据集中相应的数据的索引
        dataset = fileDataset(path_data_file,
                              transform=transform,
                              is_gray=is_gray,
                              idxs=idxs,
                              path_filter=config.save_dir + "/issue_data"
                              )
        data_list.append(dataset)

    logger.info("---生成数据统计---")
    sum_data = 0
    out = {}
    for i in range(n_clients):
        g_count = [0 for _ in range(10)]
        for c in data_list[i].labels:  #
            g_count[c] += 1
        out["generated_data_distribution"] = g_count

        # output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
        # if not os.path.exists(output_file):
        #     os.makedirs(output_file, exist_ok=True)
        logger.info("client {}, add data {}".format(i, sum(g_count)))
        sum_data += sum(g_count)
        # output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
        # with open(output_file, 'w') as w:
        #     json.dump(out, w)
    logger.info("---sum add data: {} ----".format(sum_data))



    return client_list , transform, data_list



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
    parser.add_argument("--name", type=str, default="FedMD", help="KTpFL | FedMD")
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
    parser.add_argument("--round", type=int, default=50, help="communication round")
    parser.add_argument("--local_epoch", type=int, default=1, help="local epoch")
    # 算法的特殊参数
    parser.add_argument("--public_num", type=int, default= 5000 , help="Parameters of KD-FL ")
    parser.add_argument("--Temp", type=float, default=2.0, help="Parameters of KD-FL")
    parser.add_argument("--N_alignment", type=int, default=1000, help="Parameters of KD-FL")
    parser.add_argument("--logits_matching_batchsize", type=int, default=128, help="Parameters of KD-FL")
    parser.add_argument("--min_delta", type=float, default=0.005, help="Parameters of KD-FL")
    parser.add_argument("--penalty_ratio", type=float, default=0.001, help="Parameters of KD-FL")
    parser.add_argument("--N_logits_matching_round", type=int, default=1, help="Parameters of KD-FL")
    parser.add_argument("--public_training_round", type=int, default=1, help="Parameters of KD-FL")
    # 数据增强
    parser.add_argument("--use_aug", type=str2bool,default=False, help="user dada augment or not")
    # 数据，生成数据的路径
    parser.add_argument("--gen_path", type=str, default="gen_path", help="generate image path")

    # non-IID 参数
    parser.add_argument("--noniid", type=str2bool, default=True, help="noniid or not")  # 把 True 固定了 ，type的问题，解决，添加str2bool
    # parser.add_argument("--dirichlet", type=bool, default=False, help="dirichlet or not") # 有问题
    parser.add_argument("--dirichlet",action="store_true", help="dirichlet or not")
    parser.add_argument("--alpha", type=float, default=1.0, help="control dirichlet")
    parser.add_argument("--n_cls", type=int, default=4, help="n classes per client have: 2 | 4")

    # 网络模型
    parser.add_argument("--heterogeneity", action="store_true", help="heterogeneity model")


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



def main( config, logger):

    client_list, t_transform, data_list = load_models(config, logger)     # 获取 模型对象

    # 获取公共数据,随机选取 1000个样本
    public_dset = get_public_data(config.dataset_path,
                                  config.public_num,
                                  dataname = config.dataset)  # ( n,32,32,3)

    print(type(public_dset))
    print(public_dset.shape)

    fedmd_train_pro(
                     config,
                     client_list,
                     10,                # class
                     len(client_list),  # 参与方人数
                     public_dset,
                     logger=logger,
                     gen_dataset=data_list
                 )

if __name__ == "__main__":
    new_config = init()
    logger_level = "INFO"
    logger = get_logger("col",
                        new_config.save_dir,
                        logger_level,
                        f"log_{new_config.name}.txt")  # 收为己用
    main( new_config, logger )