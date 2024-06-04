import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import RandomSampler
# 吸收
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.utils import save_image
from models.ResNet import ResNet, ResidualBlock, ResNet18


def local_train_net_scaffold( config,
                              clients,
                              global_model,
                              c_nets, c_global):
    # avg_acc = 0.0
    device = config.device

    total_delta = copy.deepcopy( global_model.state_dict() )
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, client in enumerate(clients):
        client.c_model.to(device)
        c_nets[net_id].to(device)
        c_delta_para = client.train_net_scaffold(  config,
                                                  global_model,
                                                  c_nets[net_id], c_global,
                                              )

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        # logger.info("net %d final test acc %f" % (net_id, testacc))
        # avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= config.n_clients
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    # avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)

    # nets_list = list(nets.values())
    # return nets_list

def scaffold_train(
                     config,
                    clients,
                    logger = None,
                ):
    print_fn = print if logger is None else logger.info
    print_fn(f"=================={config.name}  train ========================")
    # round = 100  #20， 200

    device = config.device
    round = config.round

    num_client = len(clients)

    in_ch = 3

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
        in_ch = 1
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
    acc_list = []

    ##################################
    # **** nets，global_model，c_nets, c_global, 都要独立

    # global_model = clients[0].c_model
    global_model = copy.deepcopy(clients[0].c_model)

    c_nets = {net_i: None for net_i in range(num_client)}
    for net_id in range(num_client):
        c_nets[net_id] = ResNet18(in_ch = in_ch)

    # c_global = global_model   # 小失误，导致不收敛
    c_global = copy.deepcopy(global_model)  # 一定得深拷贝
    c_global_para = c_global.state_dict()
    for net_id, net in c_nets.items():
        net.load_state_dict(c_global_para)

    global_para = global_model.state_dict()

    # if args.is_same_initial:
    #     for net_id, net in nets.items():
    #         net.load_state_dict(global_para)
    for idx in range(num_client):
        clients[idx].c_model.load_state_dict(global_para)

    for r in range(round):
        start_time = time.time()
        global_para = global_model.state_dict()

        # 加载模型
        for idx in range(num_client):
            clients[idx].c_model.load_state_dict(global_para)

        local_train_net_scaffold(   config,
                                   clients,
                                   global_model,
                                   c_nets, c_global,
                                )

        # update global model
        # 总数据量
        total_data_points = sum([ clients[i].num_sample for i in range(num_client) ])
        # 用户数据量的比例
        fed_avg_freqs = [ clients[i].num_sample / total_data_points for i in range(num_client) ]

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