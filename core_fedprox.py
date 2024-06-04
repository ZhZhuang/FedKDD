import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import BatchSampler
# 吸收
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.utils import save_image
from utils import get_user_data

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            #print('done')
            w_avg[k] += w[i][k]     # 有没有不能汇聚的层 ？
        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
    return w_avg


def local_train_net_fedprox( config, clients, global_model):
    global_model.to(config.device)
    for client in clients:
        client.train_net_fedprox(config,
                                global_model
                                )



def fedprox_train( config,
                clients,
                logger = None,
                ):

    print_fn = print if logger is None else logger.info
    print_fn(f"================== {config.name} train ========================")
    # round = 100  #20， 200

    round = config.round
    device = config.device
    num_client = len(clients)
    acc_list = []
    ###### 验证集
    ###### 验证集
    if config.dataset == "Cifar10":
        t_transform = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                          ])
        v_dataset = CIFAR10(root=config.dataset_path,
                            train=False,
                            download=False,
                            transform=t_transform)
    elif config.dataset == "FashionMNIST":
        t_transform = transforms.Compose([transforms.Resize(28),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])
                                          ])
        v_dataset = FashionMNIST(root=config.dataset_path,
                                 train=False,
                                 download=False,
                                 transform=t_transform)

    # v_dataset = FashionMNIST("./data", train=False, download=False, transform=transform)
    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=512,
                                               shuffle=False, num_workers=config.num_works)

    top_acc = 0.
    top_r = 0
    global_model = copy.deepcopy(clients[0].c_model)
    for r in range(round):
        start_time = time.time()
        print_fn("round : {0}".format(r))

        # 加载参数
        global_para = global_model.state_dict()
        for idx in range(num_client):
            clients[idx].c_model.load_state_dict(global_para)

        local_train_net_fedprox( config, clients, global_model)
        global_model.to('cpu')

        # 总数据量
        total_data_points = sum([clients[i].num_sample for i in range(num_client)])
        # 用户数据量的比例
        fed_avg_freqs = [clients[i].num_sample / total_data_points for i in range(num_client)]

        for idx in range(num_client):
            net_para = clients[idx].c_model.cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)
        global_model.to(device)

        # 测试
        print_fn("round {} evaluate... ".format(r))
        acc_temp = clients[0].evaluate_public(v_dataloader, global_model)
        acc_list.append(acc_temp)
        dur_time = time.time() - start_time
        print_fn("round {} mean acc {}, time: {} sec a round".format(r, acc_temp, dur_time))
        if acc_temp > top_acc:
            top_acc = acc_temp
            top_r = r

    print_fn("测试结果：")
    print_fn(acc_list)
    print_fn(f"top acc : {top_acc}, at {top_r} round!")
    print_fn(f"================== {config.name} end ========================")

