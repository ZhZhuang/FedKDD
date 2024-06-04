import copy
import logging
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST

import datasets
from data_process.data_utils import generate_alignment_data,data, TransformTwice, data_double
from torch.optim import SGD
from torch import nn
import torch.nn.functional as F
import numpy as np
# from utils.confuse_matrix import get_confuse_matrix, kd_loss

from datasets.data_utils import get_data_loader
from datasets.dataset import BasicDataset
from datasets.ssl_dataset import get_transform
# from utils import losses


def KL_loss(inputs, target, reduction='average'):
    log_likelihood = F.log_softmax(inputs, dim=1)
    #print('log_probs:',log_likelihood)
    #batch = inputs.shape[0]
    if reduction == 'average':
        #loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        loss = F.kl_div(log_likelihood, target, reduction='mean')
    else:
        #loss = torch.sum(torch.mul(log_likelihood, target))
        loss = F.kl_div(log_likelihood, target, reduction='sum')
    return loss

# 内部使用
def warm_up_simple( n ):
    if n>50:
        return 1.0
    return n / 50   # 非整除

# 以后不用了，写在client中，为 train_publicdata()
def train_one_model(model, train_dataloader, optimizer,  device, epoch,
                     Temp=1.0 ,N_round = 10,EarlyStopping=False):
    model.to(device)
    model.train()
    all_train_loss  = []
    for iter in range(epoch):
        train_loss = []
        for batch_idx, ( _, images, labels) in enumerate(train_dataloader):
            # if len(images.shape) == 4:  # 变为 b c h w
            #     images = torch.transpose(images, 3, 1)

            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            #print('images.shape:',images.shape)
            log_probs = model(images)
            #Tsoftmax = nn.Softmax(dim=1)
            #加入温度系数T#
            # if iter == 0:
            #     print('Temp:',Temp)
            output_logit = log_probs.float()/Temp
            ##
            #loss = SoftCrossEntropy(output_logit, labels)
            #loss = criterion(log_probs, labels)
            loss = KL_loss(output_logit, labels) * warm_up_simple(N_round)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        all_train_loss.append(sum(train_loss)/len(train_loss))
    return sum(all_train_loss)/len(all_train_loss)


def predict(model,dataarray,device,T):
    model.eval()
    out= []
    bs = 32
    dataarray = dataarray.astype(np.float32)
    with torch.no_grad():
        for ind in range(0,len(dataarray),bs):
            data = dataarray[ind:(ind+bs)]
            data = torch.from_numpy(data).to(device)

            logit = model(data)

            Tsoftmax = nn.Softmax(dim=1)
            #加入温度系数T#
            output_logit = Tsoftmax(logit.float()/T)

            out.append(output_logit.cpu().numpy())
    out = np.concatenate(out)
    return out

def predict_my(model,dataloader,device,T):
    model.eval()
    model.to(device)    #不写报错
    out = []
    # bs = 32
    Tsoftmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for _, x_lb, _ in dataloader:   # index , img_w, img_s
            x_lb = x_lb.to(device)
            logits = model(x_lb)
            # 加入温度系数T#
            output_logit = Tsoftmax(logits.float() / T)     # ? 蒸馏温度
            # output_logit = Tsoftmax(logits.float() )  # ? 不用蒸馏温度
            out.append(output_logit.cpu().numpy())
            # out.append(logits.cpu().numpy())
    out = np.concatenate(out)
    return out      # numpy

def val_one_model(model,dataloader,criterion=None,device= torch.device('cuda')):
    model.eval()
    acc = []
    loss_out = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            if criterion is not None:
                loss = criterion(log_probs, labels)
                loss_out.append(loss.item())
            #print('log_probs.shape:',log_probs.shape)
            acc_ = torch.mean((labels == torch.argmax(log_probs, dim=-1)).to(torch.float32))
            acc.append(acc_.item())
        if criterion is not None:
            return sum(loss_out)/len(loss_out),sum(acc)/len(acc)
        else:
            return sum(acc)/len(acc)

def get_alignment_data(public_data , N_alignment = 256):
    index = np.random.choice(range(len(public_data)), N_alignment)
    alignment_data = public_data[index] # 细节 不是()
    # alignment_data, _ = torch.utils.data.random_split(public_data, [N_alignment, len(public_data)-N_alignment])
    return alignment_data

###########
mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
mean["FashionMNIST"] = [0.286]     #计算的得到
std["FashionMNIST"] = [0.320]  #计算得到
###########

def fedmd_train(config,
               clients,
               n_classes,
               N_parties,
               public_dataset,  logger = None):

    print_fn = print if logger is None else logger.info
    print_fn("================== col train ========================")
    N_rounds = config.round
    N_private_training_round = config.local_epoch  # 默认是 1,
    N_logits_matching_round = config.N_logits_matching_round  # 1
    Temp = config.Temp  # 2
    # penalty_ratio = config.penalty_ratio  # 0.001
    num_workers = config.num_works
    logits_matching_batchsize = config.logits_matching_batchsize  #
    batch_size = config.batch_size
    N_alignment = config.N_alignment
    public_training_round = config.public_training_round
    device = torch.device(config.device)
    performance_public = {i: [] for i in range(N_parties)}  # 公共分布的表现
    performance_self = {i: [] for i in range(N_parties)}    # 自身分布的表现
    r = 0

    is_gray = False
    if config.dataset != "Cifar10":
        is_gray = True

    if config.dataset == "Cifar10":
        crop_size = 32  # cifar 10 图片的尺寸
        transform = get_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], crop_size, train=True)  # cifar 10
        v_transform = get_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], crop_size, train=False)  # cifar 10
        v_dataset = CIFAR10(root=config.dataset_path,
                            train=False,
                            download=False,
                            transform=v_transform)
    elif config.dataset == "FashionMNIST":
        crop_size = 28  # fashionMNIST 图片的尺寸
        transform = get_transform(0.5, 0.5, crop_size, train=True)  # cifar 10
        v_transform = get_transform(0.5, 0.5, crop_size, train=False)  # cifar 10
        v_dataset = FashionMNIST(root=config.dataset_path,
                                 train=False,
                                 download=False,
                                 transform=v_transform)

    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=1028, shuffle=True)

    top_acc = 0.0
    top_r = 0
    acc_list = []   #平均精度

    top_self_acc = 0.0
    top_self_r = 0
    acc_self_list = []  # 自身数据集测试的

    while True:
        start_time = time.time()

        # At beginning of each round, generate new alignment dataset
        # 注意 ： 共有数据集可以是无标签数据
        alignment_data = get_alignment_data(public_data=public_dataset, N_alignment=N_alignment)
        alignment_dataset = BasicDataset("pimodel",  # 返回 index， img_w, img_s
                       alignment_data,
                       None,# target
                       n_classes,
                       transform,
                       True,  # is_ulb 无标签数据
                       None,  # strong_transform
                       onehot=False)  # 默认都是非独热的
        # alignment_dataset = get_alignment_data(public_data= pre_public_dataset, N_alignment=N_alignment)
        alignment_dataloader = get_data_loader(alignment_dataset,
                                            logits_matching_batchsize,     # batch_size    默认 predict的默认偏移 32
                                            num_workers=8,
                                            drop_last=False)

        print_fn(f"round {r}")
        print("update logits ... ")
        # update logits
        logits = []
        #各个用户对共有数据进行预测
        for client in clients:       #model
            logits.append(predict_my(client.c_model, alignment_dataloader, device, Temp))  # 只是对共有数据的预测， 没有训练

        # fedmd 直接取平均
        ################   核心操作   ################
        logits = np.sum(logits, axis=0)     #
        print('logits.shape:', logits.shape)
        logits /= N_parties

        # ################ 测试模型 ###############
        print("test performance ... ")
        total_acc = 0.0
        total_self_acc = 0.0
        for index, client in enumerate(clients):
            # acc  = client.evaluate( v_dataloader )  # 每个模型对私有自身私有数据的评测, 此步前必须经过train()
            acc = client.evaluate(v_dataloader)  # 测试公共数据
            self_acc = client.evaluate()  # 测试自身数据集
            performance_public[index].append(acc)
            performance_self[index].append(self_acc)
            # print("collaboration_performance[index][-1] ",collaboration_performance[index][-1])
            print_fn('round:{}, model:{},public acc:{}, self acc {}'.format(r, index, acc, self_acc))
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

        r += 1
        if r > N_rounds :
            break

        print("updates models ...")
        for index, client in enumerate( clients ):
            ###################        公有数据全部用户都需要训练           ###########################
            print("model {0} starting alignment with public logits... ".format(index))
            public_predict_dset = datasets.BasicDataset.BasicDataset(alignment_data,
                                                                     targets= logits,
                                                                     transform=transform,
                                                                     is_gray=is_gray,
                                                                     onehot=False)

            public_predict_train_loader = get_data_loader( public_predict_dset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=False)

            # 各个参与方 对共有数据 和 其预测 进行训练
            for i in range(public_training_round):
                client.train_publicdata(
                    public_predict_train_loader,
                    device,
                    N_logits_matching_round,
                    Temp=Temp
                )

            print("model {0} done public data training！".format(index))
            ###################       end 公有数据全部用户都需要训练           ###########################

            ###################        用户私有数据进行训练           ###########################
            # for index , client in enumerate(clients):
            print("model {0} starting training with private data... ".format(index))
            for i in range(N_private_training_round):   # 1 、 10
                client.train_classifier(config)
            print("model {0} done private training. \n".format(index))
            ###################       end 用户私有数据进行训练          ###########################

        dur_time = time.time() - start_time
        print_fn("round {} , time: {} sec a round".format(r, dur_time))
        # END FOR LOOP

    # END WHILE LOOP
    # print_fn("#########  公共分布 训练结果   #########")
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


