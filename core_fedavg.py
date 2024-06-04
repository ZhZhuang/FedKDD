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

# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             #print('done')
#             w_avg[k] += w[i][k]     # 有没有不能汇聚的层 ？
#         # w_avg[k] = torch.div(w_avg[k], len(w))
#         w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
#     return w_avg

def FedAvg(w, wl= None):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0 :
                w_avg[k] = w_avg[k] * wl[i]
            #print('done')
            else:
                w_avg[k] += w[i][k] * wl[i]     # 有没有不能汇聚的层 ？
        # w_avg[k] = torch.div(w_avg[k], len(w))
        # w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
    return w_avg

def fedavg_train( config,
                    clients,
                    logger = None
                ):
    # 测试环境
    print(torch.cuda.is_available())  # 查看CUDA是否可用
    print(torch.cuda.device_count())  # 查看可用的CUDA数量
    print(torch.version.cuda)  # 查看CUDA的版本号

    print_fn = print if logger is None else logger.info
    print_fn(f"================== {config.name} train ========================")
    # round = 100  #20， 200
    round = config.round  # 100,
    num_client = len(clients)

    w_locals = [ clients[0].c_model.state_dict() for i in range(len(clients)) ] #也可以b = [0]*10

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
                                               shuffle=False, num_workers= config.num_works)

    top_acc = 0.
    top_r = 0
    acc_list = []

    top_self_acc = 0.0
    top_self_r = 0
    acc_self_list = []  # 自身数据集测试的

    # start collaborating training
    performance_public = {i: [] for i in range(num_client)}  # 公共分布的表现
    performance_self = {i: [] for i in range(num_client)}  # 自身分布的表现

    for r in range(round):
        start_time = time.time()
        print_fn("round : {0}".format(r))

        # 本地训练
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)    # 选择部分用户
        for index, client in enumerate(clients):
            client.train_classifier( config )  # 对自身私有数据集训练一遍
            # client.train_classifier_1(dataloader_list[i])  #
            # client.train_classifier(t_dataloader)  # 效果最差的方式， 同事训练自身的数据和 生成数据
            w_locals[index] = copy.deepcopy( client.c_model.state_dict() )

        # 总数据量
        total_data_points = sum([clients[i].num_sample for i in range(num_client)])
        # 用户数据量的比例
        fed_avg_freqs = [clients[i].num_sample / total_data_points for i in range(num_client)]

        # # 联邦聚合
        with torch.no_grad():
            w_global = FedAvg(w_locals, fed_avg_freqs)
        #     # w_optim_global = FedAvg(w_optims)
        for index in range(len(clients)):
            clients[index].c_model.load_state_dict(w_global,strict = True)  # 适合部分加载模型


        #  ============= 测试 ===========
        # print_fn("round {} evaluate... ".format(r))
        # # for index, client in enumerate(clients):
        # #     # print("model {0} evaluate... ".format(index))
        # #     acc_list[index] = client.evaluate(v_dataloader)  # 对自身私有数据集训练一遍
        # acc_temp =clients[0].evaluate_public(v_dataloader)
        # acc_list.append( acc_temp )
        # dur_time = time.time() - start_time
        # print_fn("round {} mean acc {}, time: {} sec a round".format(r, acc_temp, dur_time))
        # if acc_temp > top_acc:
        #     top_acc = acc_temp
        #     top_r = r
        # clients[0].evaluate(v_dataloader)

        # ################ 测试模型 ###############
        print("test performance ... ")
        total_acc = 0.0
        total_self_acc = 0.0
        for index, client in enumerate(clients):
            # acc  = client.evaluate( v_dataloader )  # 每个模型对私有自身私有数据的评测, 此步前必须经过train()
            acc = client.evaluate(v_dataloader)  # 测试公共分布数据集
            self_acc = client.evaluate()  # 测试自身数据集
            performance_public[index].append(acc)
            performance_self[index].append(self_acc)
            # print("collaboration_performance[index][-1] ",collaboration_performance[index][-1])
            print_fn('round:{}, model:{}, public acc:{}, self acc {}'.format(r, index, acc, self_acc))
            total_acc += acc
            total_self_acc += self_acc

        avg_acc = total_acc / (len(clients))
        if avg_acc > top_acc:
            top_acc = avg_acc
            top_r = r

        avg_self_acc = total_self_acc / (len(clients))
        if avg_self_acc > top_self_acc:
            top_self_acc = avg_self_acc
            top_self_r = r

        print_fn('round:{} public avg acc {}, self avg acc {}'.format(r, avg_acc, avg_self_acc))

        acc_list.append(avg_acc)
        acc_self_list.append(avg_self_acc)

        dur_time = time.time() - start_time
        print_fn("round {} , time: {} sec a round".format(r, dur_time))

    # print_fn("全局测试结果：")
    # print_fn(acc_list)
    # print_fn(f"top acc : {top_acc}, at {top_r} round!")
    # print_fn(f"================== {config.name} end ========================")

    # END WHILE LOOP
    print_fn("publice distribution results")
    for i in performance_public:  # 集合
        print_fn(f"clients {i} preformance is :{performance_public[i]}")

    print_fn("self distribution results")
    for i in performance_public:  # 集合
        print_fn(f"clients {i} preformance is :{performance_self[i]}")

    print_fn("测试集公共分布平均测试结果：")
    print_fn(acc_list)
    print_fn(f"top acc : {top_acc}, at {top_r} round!")

    print_fn("测试集自身分布平均测试结果：")
    print_fn(acc_self_list)
    print_fn(f"top acc : {top_self_acc}, at {top_self_r} round!")
    return performance_public


