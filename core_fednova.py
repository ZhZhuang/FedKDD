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

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            #print('done')
            w_avg[k] += w[i][k]     # 有没有不能汇聚的层 ？
        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
    return w_avg

def local_train_net_fednova( config, clients,  global_model):
    # avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(config.device)
    for client in clients:
        # trainacc, testacc,\
        a_i, d_i = client.train_net_fednova( config,
                                            global_model
                                          )

        a_list.append(a_i)
        d_list.append(d_i)
        # n_i = len(train_dl_local)
        n_i = len(client.dataloader)
        n_list.append(n_i)
        # print("net %d final test acc %f" % (net_id, testacc))
        # avg_acc += testacc


    # avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     print("avg test acc %f" % avg_acc)

    # nets_list = list(nets.values())
    # return nets_list, a_list, d_list, n_list
    return  a_list, d_list, n_list

def fednova_train( config,
                clients,
                logger = None,
                ):

    print_fn = print if logger is None else logger.info
    print_fn(f"================== {config.name} train ========================")
    # round = 100  #20， 200

    device = config.device
    round = config.round

    # # 总数据量
    # data_sum = 0
    # # 每个用户的数据比例
    # portion = []

    num_client = len(clients)

    # global_model = clients[0].c_model
    global_model = copy.deepcopy(clients[0].c_model)

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
    acc_list = []

    for r in range(round):
        start_time = time.time()
        logger.info("in comm round:" + str(r))

        global_para = global_model.state_dict()

        # 加载模型
        for idx in range(num_client):
            clients[idx].c_model.load_state_dict(global_para)

        # 本地 训练
        a_list, d_list, n_list = local_train_net_fednova(  config,
                                                          clients,
                                                          global_model
                                                        )
        total_n = sum(n_list)
        # print("total_n:", total_n)
        d_total_round = copy.deepcopy(global_model.state_dict())
        for key in d_total_round:
            d_total_round[key] = 0.0

        for i in range(num_client):
            d_para = d_list[i]
            for key in d_para:
                # if d_total_round[key].type == 'torch.LongTensor':
                #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                # else:
                d_total_round[key] += d_para[key] * n_list[i] / total_n

        # update global model
        coeff = 0.0
        for i in range(num_client):
            coeff = coeff + a_list[i] * n_list[i] / total_n

        updated_model = global_model.state_dict()
        for key in updated_model:
            # print(updated_model[key])
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                # print(updated_model[key].type())
                # print((coeff*d_total_round[key].type()))
                updated_model[key] -= coeff * d_total_round[key]
        global_model.load_state_dict(updated_model)

        # logger.info('global n_training: %d' % len(train_dl_global))
        # logger.info('global n_test: %d' % len(test_dl_global))

        global_model.to(device)
        # train_acc = compute_accuracy(global_model, train_dl_global, device=device)
        # test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

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

