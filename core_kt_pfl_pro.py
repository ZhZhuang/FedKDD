import copy
import logging
import sys
import time

import torch
from torch.utils.data import DataLoader
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
import datasets
# from utils import losses


######################         core         ###############################
def get_models_logits(raw_logits, weight_alpha, N_models, penalty_ratio):  # raw_logits为list-np；weight为tensor；
    weight_mean = torch.ones(N_models, N_models, requires_grad=True)
    weight_mean = weight_mean.float() / (N_models)
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)  # reduction的意思是维度要不要缩减，以及怎么缩减
    teacher_logits = torch.zeros(N_models, np.size(raw_logits[0], 0), np.size(raw_logits[0], 1),
                                 requires_grad=False)  # 创建logits of teacher  #next false
    models_logits = torch.zeros(N_models, np.size(raw_logits[0], 0), np.size(raw_logits[0], 1),
                                requires_grad=True)  # 创建logits of teacher
    # weight.requires_grad = True #can not change requires_grad here
    weight = weight_alpha.clone()
    for self_idx in range(N_models):  # 对每个model计算其teacher的logits加权平均值
        teacher_logits_local = teacher_logits[self_idx]
        for teacher_idx in range(N_models):  # 对某一model，计算其他所有model的logits
            # if self_idx == teacher_idx:
            #     continue
            # teacher_tmp = weight[self_idx][teacher_idx] * raw_logits[teacher_idx]
            # teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * raw_logits[teacher_idx])
            # teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * torch.autograd.Variable(torch.from_numpy(raw_logits[teacher_idx])))
            # teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx]))
            teacher_logits_local = torch.add(teacher_logits_local,
                                             weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx]))
            #                                                                tensor中的一个像素点，本质标量 * teacher的完整logits

        loss_input = torch.from_numpy(raw_logits[self_idx])
        # loss_target = torch.autograd.Variable(teacher_logits[self_idx], requires_grad=True)
        loss_target = teacher_logits_local

        loss = loss_fn(loss_input, loss_target)

        loss_penalty = loss_fn(weight[self_idx], weight_mean[self_idx])
        # print('loss_penalty:', loss_penalty)
        # print('loss:', loss)
        # loss += loss_penalty*penalty_ratio
        loss = loss + loss_penalty * penalty_ratio
        # loss = SoftCrossEntropy_without_logsoftmax(loss_input,loss_target)

        # weight[self_idx].zero_grad()
        # weight[self_idx].grad.zero_()
        weight.retain_grad()  # 保留叶子张量grad
        # print('weight.grad before loss.backward:', weight.grad)
        loss.backward(retain_graph=True)
        # print('weight:', weight)

        # print('weight.requires_grad:', weight.requires_grad)
        # print('weight.grad:', weight.grad)
        # print('weight[self_idx]:', weight[self_idx])
        # print('weight[self_idx].grad:', weight[self_idx].grad)
        with torch.no_grad():
            # weight[self_idx] = weight[self_idx] - weight[self_idx].grad * 0.001  #更新权重
            gradabs = torch.abs(weight.grad)
            gradsum = torch.sum(gradabs)
            gradavg = gradsum.item() / (N_models)
            grad_lr = 1.0
            for i in range(5):  # 0.1
                if gradavg > 0.01:
                    gradavg = gradavg * 1.0 / 5
                    grad_lr = grad_lr / 5
                if gradavg < 0.01:
                    gradavg = gradavg * 1.0 * 5
                    grad_lr = grad_lr * 5
            # print('grad_lr:', grad_lr)
            weight.sub_(weight.grad * grad_lr)
            # weight.sub_(weight.grad*50)
            weight.grad.zero_()
    #############设定权重######################
    # set_weight_local = []
    # weight1 = [0.18, 0.18, 0.18, 0.18, 0.18, 0.02, 0.02, 0.02, 0.02, 0.02]
    # weight2 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.18, 0.18, 0.18, 0.18, 0.18]
    # for i in range(N_models):
    #     if i <= 4:
    #         set_weight_local.append(weight1)
    #     if i >= 5:
    #         set_weight_local.append(weight2)
    # tensor_set_weight_local = torch.Tensor(set_weight_local)
    ###################################
    # 更新 raw_logits
    for self_idx in range(N_models):  # 对每个model计算其teacher的logits加权平均值
        weight_tmp = torch.zeros(N_models)
        idx_count = 0
        for teacher_idx in range(N_models):  # 对某一model，计算其softmax后的weight
            # if self_idx == teacher_idx:
            #     continue
            # weight加softmax#
            weight_tmp[idx_count] = weight[self_idx][teacher_idx]
            idx_count += 1
        # softmax_fn = nn.softmax() #这里不对，不应该softmax，应该normalization##先用低温softmax#
        weight_local = nn.functional.softmax(weight_tmp * 5.0)

        idx_count = 0
        for teacher_idx in range(N_models):  # 对某一model，计算其他所有model的logits
            # if self_idx == teacher_idx:
            #     continue
            # models_logits[self_idx] = torch.add(models_logits[self_idx], weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx]))
            # 设定权重 models_logits[self_idx] = torch.add(models_logits[self_idx], tensor_set_weight_local[self_idx][idx_count] * torch.from_numpy(raw_logits[teacher_idx]))
            # models_logits[self_idx] = torch.add(models_logits[self_idx], weight_local[idx_count] * torch.from_numpy(raw_logits[teacher_idx])) # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation
            # models_logits[self_idx] = models_logits[self_idx]+weight_local[idx_count] * torch.from_numpy(raw_logits[teacher_idx])
            with torch.no_grad():
                # 设定权重weight[self_idx][teacher_idx] = tensor_set_weight_local[self_idx][idx_count]
                models_logits[self_idx] = torch.add(models_logits[self_idx],
                                                    weight_local[idx_count] * torch.from_numpy(raw_logits[teacher_idx]))
                weight[self_idx][teacher_idx] = weight_local[idx_count]
            idx_count += 1
    # print('weight after softmax:', weight)
    return models_logits, weight

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

def predict_my(model,dataloader,device,T = 2.0):
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
    # print(type(index))
    # print(index)
    alignment_data = public_data[index] # 细节 不是()
    # alignment_data, _ = torch.utils.data.random_split(public_data, [N_alignment, len(public_data)-N_alignment])
    return alignment_data

###########
# mean, std = {}, {}
# mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
# std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
# mean["FashionMNIST"] = [0.28604063]     #计算的得到
# std["FashionMNIST"] = [0.32045463]  #计算得到
###########

def kt_pfl_train_pro(config,
               clients,
               n_classes,
               N_parties,
               public_dataset,
               logger = None,
               gen_dataset = None):

    print_fn = print if logger is None else logger.info
    print_fn("================== col train ========================")
    # print_fn(KT_pFL_params)

    N_rounds = config.round
    N_private_training_round = config.local_epoch  # 默认是 1,
    N_logits_matching_round = config.N_logits_matching_round      # 1
    public_training_round = config.public_training_round
    Temp = config.Temp    # 2
    penalty_ratio = config.penalty_ratio  # 0.001
    num_workers = config.num_works
    logits_matching_batchsize = config.logits_matching_batchsize #
    batch_size = config.batch_size
    N_alignment = config.N_alignment
    device = torch.device(config.device)
    # start collaborating training
    performance_public = {i: [] for i in range(N_parties)}  # 公共分布的表现
    performance_self = {i: [] for i in range(N_parties)}  # 自身分布的表现

    r = 0
    weight_alpha = torch.ones(N_parties, N_parties, requires_grad=True)  # eight_alpha  is c in paper
    weight_alpha = weight_alpha.float() / (N_parties)

    is_gray = False
    if config.dataset != "Cifar10":
        is_gray = True

    if config.dataset == "Cifar10":
        crop_size = 32  # cifar 10 图片的尺寸
        transform = get_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], crop_size, train= True)  # cifar 10
        v_transform = get_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], crop_size, train= False)  # cifar 10
        v_dataset = CIFAR10(root=config.dataset_path,
                            train=False,
                            download=False,
                            transform=v_transform)
    elif config.dataset == "FashionMNIST":
        crop_size = 28  # fashionMNIST 图片的尺寸
        transform = get_transform(0.5, 0.5, crop_size, train= True)  # cifar 10
        v_transform = get_transform(0.5, 0.5, crop_size, train=False)  # cifar 10
        v_dataset = FashionMNIST(root=config.dataset_path,
                            train=False,
                            download=False,
                            transform=v_transform)

    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size= 1028, shuffle=True)

    top_acc = 0.0
    top_r = 0
    acc_list = []

    top_self_acc = 0.0
    top_self_r = 0
    acc_self_list = []  # 自身数据集测试的

    while True:
        start_time = time.time()
        # At beginning of each round, generate new alignment dataset
        # 注意 ： 共有数据集可以是无标签数据
        alignment_data = get_alignment_data(public_data=public_dataset, N_alignment=N_alignment)
        alignment_dataset = BasicDataset("pimodel",  # 返回 index， img_w, img_s # “pimodel”是代码的历史遗留问题
                                         alignment_data,
                                         None,  # target
                                         n_classes,
                                         transform,
                                         True,  # is_ulb 无标签数据
                                         None,  # strong_transform
                                         onehot=False)  # 默认都是非独热的
        # alignment_dataset = get_alignment_data(public_data= pre_public_dataset, N_alignment=N_alignment)
        alignment_dataloader = get_data_loader(alignment_dataset,
                                               logits_matching_batchsize,  # batch_size    默认 predict的默认偏移 32
                                               num_workers=8,
                                               drop_last=False)

        print_fn(f"round {r}")
        print("update logits ... ")
        # update logits
        logits = []
        # 各个用户对共有数据进行预测
        for client in clients:  # model
            logits.append(predict_my(client.c_model, alignment_dataloader, device, Temp))  # 只是对共有数据的预测， 没有训练

        # logits_models is predict of per model
        logits_models, weight_alpha = get_models_logits(logits, weight_alpha, N_parties, penalty_ratio)  # 核心
        logits_models = logits_models.detach().numpy()

        # ################ 测试模型 ###############
        print("test performance ... ")
        total_acc = 0.0
        total_self_acc = 0.0
        for index, client in enumerate(clients):
            # acc  = client.evaluate( v_dataloader )  # 每个模型对私有自身私有数据的评测, 此步前必须经过train()
            acc = client.evaluate(v_dataloader)     # 测试公共分布数据集
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

        r += 1
        if r > N_rounds:
            break

        print("updates models ...")

        for index, client in enumerate(clients):
            ###################        公有数据全部用户都需要训练           ###########################
            print("model {0} starting alignment with public logits... ".format(index))
            # logits_models is predict of aggregation for public dataset
            # 得到公共数据集的 dataloader = public_dataset.data + logits_models[index],

            public_predict_dset = datasets.BasicDataset.BasicDataset(alignment_data,
                                               targets=logits_models[index],
                                               transform=transform,
                                               is_gray=is_gray,
                                               onehot=False)

            public_predict_train_loader = get_data_loader(public_predict_dset,
                                                          batch_size=batch_size,  # batch_size 64
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

            print("model {0} done alignment".format(index))
            ###################       end 公有数据全部用户都需要训练           ###########################

            ###################        用户私有数据进行训练           ###########################
            print("model {0} starting training with private data... ".format(index))
            for i in range(N_private_training_round):  # 1 、 10
                # client.train(args_list[index])  # 对自身私有数据集训练一遍
                # client.train_classifier_1()  # 对自身私有数据集训练一遍
                train_dataset = gen_dataset[i] + client.dataset     # 混合 生成 和 私有 数据
                train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=config.batch_size,
                                                           num_workers=config.num_works, shuffle=True)
                # client.train_classifier( config ) # 使用自身数据
                client.train_classifier_2(train_dataloader)  # !!
            print("model {0} done private training. \n".format(index))
            ###################       end 用户私有数据进行训练          ###########################
        # END FOR LOOP

        dur_time = time.time() - start_time
        print_fn("round {} , time: {} sec a round".format(r, dur_time))


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
