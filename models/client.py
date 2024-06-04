import json
import os
from collections import Counter
from copy import deepcopy

import numpy
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
# from cleanlab.filter import find_label_issues, find_label_issues_using_argmax_confusion_matrix  #

# ********************  label preprocess  ******************************** #
onehot = torch.zeros(10, 10)
# 产生 0 到 9 的独热向量    ，没有完全理解scatter
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
# fill = torch.zeros([10, 10, opt.img_size, opt.img_size])
fill = torch.zeros([10, 10, 32, 32])
# 核心操作
# 将 10 * 10 * 32 * 32 的全零向量的 第 i 项的 第i通道全部设置为 1
for i in range(10):
    fill[i, i, :, :] = 1
# ********************  label preprocess end ******************************** #

cuda = True
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
adversarial_loss = torch.nn.MSELoss().cuda() if cuda else torch.nn.MSELoss()



class Client:
    def __init__(self,
                 config,
                 C_model= None, client_idx=0,
                 #model=None,
                 num_classes= 10,
                 dataloader = None,
                 t_dataloader = None,
                 dataset = None,
                 logger = None):
        # self.model =model       # python 的传值方式， ！！ 对象引用计数器 ？？

        self.c_model = C_model.cuda()
        self.num_classes = num_classes
        self.print_fn = print if logger is None else logger.info  # is 判断地址， == 判断的是值
        self.dataloader = dataloader
        self.dataset = dataset
        self.test_dataloader = t_dataloader
        self.idx= client_idx
        self.all_class = [] # 存放所有包含的类别
        self.iter = 0
        self.num_sample = len(dataset)
        self.criterion = torch.nn.CrossEntropyLoss()


        dist_file_name = f"{config.save_dir}/client_data_statistics_{client_idx}.json"
        with open(dist_file_name, 'r') as f:
            content = json.loads(f.read())
        temp = content['distribution']
        for i, elemnt in enumerate(temp):
            if elemnt != 0:
                self.all_class.append(i)
            # else:
            #     self.all_class.append(i)

    def set_optimizer(self,opti_C = None ,opti_G = None, opti_D = None):
        if opti_G != None and opti_D != None:
            self.optimizer_G = opti_G
            self.optimizer_D = opti_D
        self.optimizer_C = opti_C

    def set_gan_model(self,G_model , D_model ):
        self.generator = G_model.cuda()
        self.discriminator = D_model.cuda()

    def set_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def train_gan(self):
        for i, (imgs, labels) in enumerate(self.dataloader):
            batch_size = imgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)  # pytorch tensor的操作
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            # labels = Variable(labels.type(LongTensor))
            labels = labels.type(LongTensor)

            # -----------------
            #  Train Generator
            # -----------------

            self.optimizer_G.zero_grad()
            z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
            # 生成的标签应该只在样本包含的标签中
            y_ = torch.Tensor(numpy.random.choice(self.all_class, size=batch_size, replace=True)).type(torch.LongTensor)
            # y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()  # ？rand（）产生0~1之间的数，squeeze去除1维度
            y_label_ = onehot[y_]  # 生成器  10 * 1 * 1， 此处可以不用数组，直接利用onehot()函数
            y_fill_ = fill[y_]  # 用于判别器 10 * 32 * 32  ，其中一个通道全为 1
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

            # Generate a batch of images
            # gen_imgs = generator(z_, gen_labels)
            gen_imgs = self.generator(z_, y_label_)

            # Loss measures generator's ability to fool the discriminator
            # validity = discriminator(gen_imgs, gen_labels)
            validity = self.discriminator(gen_imgs, y_fill_)
            g_loss = adversarial_loss(validity, valid)  # ？？？  判别的值和 1 作损失

            g_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Loss for real images
            # validity_real = discriminator(real_imgs, labels)
            labels_fill = fill[labels]
            # labels_fill = Variable(labels_fill) # error ,因为fill 在cpu中
            labels_fill = Variable(labels_fill.cuda())
            validity_real = self.discriminator(real_imgs, labels_fill)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            # validity_fake = discriminator(gen_imgs.detach(), gen_labels)    #？ detach 返回一个新的tensor，从当前计算图中分离下来。但是仍指向原变量的存放位置，不具有grad
            validity_fake = self.discriminator(gen_imgs.detach(), y_fill_)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            if i % 60 == 0:
                self.print_fn(
                    "[Client %d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (self.idx, i, len(self.dataloader), d_loss.item(), g_loss.item())
                )

    def train_publicdata(self, train_dataloader, device, epoch,
                         Temp=1.0, beta = 1.0):
        model = self.c_model
        model.cuda()
        model.train()

        # scaler = GradScaler()
        # amp_cm = autocast if args.amp else contextlib.nullcontext  # 细节

        all_train_loss = []
        for iter in range(epoch):
            train_loss = []
            # for batch_idx, (images, labels, _) in enumerate(train_dataloader):
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)
                # with amp_cm():
                model.zero_grad()

                log_probs = model(images)

                # 可以加上softmax
                output_logit = log_probs.float() / Temp  # 可有，可无
                # 注意，此处，本地可以使用交叉熵损失，超算只能用kl损失
                #此处同一改为KL损失
                loss = KL_loss(output_logit, labels)
                # labels = labels.long()  # 不加，超算上报错  ！！！！！！
                # loss = self.criterion(output_logit, labels)
                loss.backward()
                self.optimizer_C.step()
                train_loss.append(loss.item())
            all_train_loss.append(sum(train_loss) / len(train_loss))
        return sum(all_train_loss) / len(all_train_loss)

    def train_classifier(self, config):
        self.c_model.train()
        for i in range(config.local_epoch):  # 1 、 10\ 5
            # for i, (imgs, labels, _) in enumerate(self.dataloader):
            for i, (imgs, labels) in enumerate(self.dataloader):
            # for (imgs, labels), (gen_imgs, gen_labels) in zip(self.dataloader, Dataloader):
                imgs, labels = imgs.cuda(), labels.cuda()
                # gen_imgs, gen_labels = gen_imgs.cuda(), gen_labels.cuda()
                # batch_size = imgs.shape[0]
                logits = self.c_model(imgs)
                real_loss = self.criterion(logits, labels)  # ce_loss 原理 !!
                total_loss = real_loss.cuda()   # + 0.5 * gen_loss
                total_loss.backward()
                self.optimizer_C.step()
                self.c_model.zero_grad()
                self.iter += 1

    def train_classifier_2(self, Dataloader=None):
        self.c_model.train()
        dataloader = Dataloader
        for i, (imgs, labels ) in enumerate(dataloader):
            # for (imgs, labels), (gen_imgs, gen_labels) in zip(self.dataloader, Dataloader):
            imgs, labels = imgs.cuda(), labels.cuda()
            # gen_imgs, gen_labels = gen_imgs.cuda(), gen_labels.cuda()
            # batch_size = imgs.shape[0]
            logits = self.c_model(imgs)
            # real_loss = ce_loss(logits, labels, reduction='mean')  # ce_loss 原理 !!
            real_loss = self.criterion(logits, labels)  # ce_loss 原理 !!
            # gen_loss = ce_loss(logits_gen[mask], gen_labels[mask], reduction="mean")  # 选择生成样本的一部分
            # total_loss = real_loss.cuda() + 0.5 * gen_loss
            total_loss = real_loss.cuda()  # + 0.5 * gen_loss
            total_loss.backward()
            self.optimizer_C.step()
            self.c_model.zero_grad()
            # self.print_fn(f"classifier loss: %0.2f")
            # if i % 60 == 0:
            #     self.print_fn(
            #         "[Client %d] [Batch %d/%d] [classifier loss: %f]"
            #         % (self.idx, i, len(self.dataloader), total_loss.item())
            #     )
        self.iter += 1

    def train_classifier_temp(self, Dataloader=None, user_lr_scheduler= False):
        self.c_model.train()
        dataloader = Dataloader
        for i, (imgs, labels) in enumerate(dataloader):
            # for (imgs, labels), (gen_imgs, gen_labels) in zip(self.dataloader, Dataloader):
            imgs, labels = imgs.cuda(), labels.cuda()
            # gen_imgs, gen_labels = gen_imgs.cuda(), gen_labels.cuda()
            # batch_size = imgs.shape[0]
            logits = self.c_model(imgs)
            # real_loss = ce_loss(logits, labels, reduction='mean')  # ce_loss 原理 !!
            real_loss = self.criterion(logits, labels)  # ce_loss 原理 !!
            # gen_loss = ce_loss(logits_gen[mask], gen_labels[mask], reduction="mean")  # 选择生成样本的一部分
            # total_loss = real_loss.cuda() + 0.5 * gen_loss
            total_loss = real_loss.cuda()  # + 0.5 * gen_loss
            total_loss.backward()
            self.optimizer_C.step()

            self.c_model.zero_grad()
            # self.print_fn(f"classifier loss: %0.2f")
            # if i % 60 == 0:
            #     self.print_fn(
            #         "[Client %d] [Batch %d/%d] [classifier loss: %f]"
            #         % (self.idx, i, len(self.dataloader), total_loss.item())
            #     )
        if user_lr_scheduler:
            self.lr_scheduler.step()
        self.iter += 1

    # def train_classifier(self, Dataloader = None):
    #     self.c_model.train()
    #     # if Dataloader is None:  # 多浪费了一句代码， 参看evaluate()
    #     #     dataloader = self.dataloader
    #     # else:
    #     #     dataloader = Dataloader
    #     # for i, (imgs, labels) in enumerate(dataloader):
    #     for (imgs, labels, _), (gen_imgs, gen_labels, _) in zip(self.dataloader, Dataloader):
    #         imgs , labels = imgs.cuda(), labels.cuda()
    #         gen_imgs, gen_labels = gen_imgs.cuda(), gen_labels.cuda()
    #         # batch_size = imgs.shape[0]
    #         logits = self.c_model(imgs)
    #         # logits = F.softmax(logits, dim= 1)    # 加的话，影响后续梯度计算，无法正常训练
    #         logits_gen = self.c_model(gen_imgs)
    #         # logits_gen = F.softmax(logits_gen, dim= 1)
    #         # if self.iter > 10:
    #         #     self.print_fn("------------- logits -----------")
    #         #     self.print_fn(logits)
    #         #     self.print_fn("------------- logits_gen -----------")
    #         #     self.print_fn(logits_gen)
    #
    #         # 改进：不使用全部的生成图片，只选择 正向推理大于阈值的生成样本
    #         #      可以接着改进，利用 flexmatch中的课程伪标签的思想
    #         # 问题：生成图片质量太低，有些类质量太差！！，改进GAN网络
    #
    #         # ！！！！！！！！
    #         # 对于梯度更新还是没有全理解， 分析 logits_gen = self.c_model(gen_imgs)放到 with中
    #         with torch.no_grad():       # with 语法
    #             logits_gen = F.softmax(logits_gen, dim=1)
    #             preds, _ = torch.max(logits_gen, dim=1)
    #             # 调整
    #             confidence_mask = (preds >= 0.6)   # 获取置信度较高的 样本 0.6 0.8(max)  0.7 0.9
    #
    #             # uncertainty = -1.0 * torch.sum(logits_gen * torch.log(logits_gen + 1e-6), dim=1)  # 信息熵
    #             # uncertainty_mask = (uncertainty < 1.0)    # 2.0 4.0, 1.0  感觉没有意义, 经验证,无意义
    #             mask = confidence_mask  # * uncertainty_mask #
    #         real_loss = self.criterion(logits, labels)       # ce_loss 原理 !!
    #         gen_loss = self.criterion(logits_gen[mask], gen_labels[mask])   # 选择生成样本的一部分
    #         # total_loss = real_loss.cuda() + 0.5 * gen_loss
    #         # 使用加热函数
    #         total_loss = real_loss.cuda() + 0.5 * gen_loss  # 0.3 0.5 0.7(max)  0.9 0.8
    #         total_loss.backward()       # 回顾 权重的更新 !!!!!!!!!!!
    #         self.optimizer_C.step()
    #         self.c_model.zero_grad()
    #         # self.print_fn(f"classifier loss: %0.2f")
    #         # if i % 60 == 0:
    #         #     self.print_fn(
    #         #         "[Client %d] [Batch %d/%d] [classifier loss: %f]"
    #         #         % (self.idx, i, len(self.dataloader), total_loss.item())
    #         #     )
    #     self.iter += 1

    # 通过置信学习开源工具：cleanlab,过滤生成样本
    @torch.no_grad()
    def filter_with_cleanlab(self, dataloader ):
        drop_list = []
        all_labels = np.array([]).astype(int)
        all_idxs = np.array([])
        all_pred_probs = None
        for i, (images, labels, idxs) in enumerate(dataloader):
            # bs = images.size(0)
            # print(bs)
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass.
            outputs = self.c_model(images)
            preds = F.softmax(outputs, dim=1)
            # preds, _ = torch.max(preds, dim=1)

            labels = labels.cpu().numpy()
            pred_probs = preds.cpu().numpy()
            idxs = idxs.cpu().numpy()
            # 放到一个大数组中存储
            all_labels = np.concatenate( (all_labels, labels), axis=0)
            if i == 0:
                all_pred_probs = pred_probs
            else:
                all_pred_probs = np.concatenate( (all_pred_probs, pred_probs), axis=0)
            all_idxs = np.concatenate( (all_idxs, idxs), axis=0)

        ordered_label_issues = find_label_issues(
            labels=all_labels,
            pred_probs=all_pred_probs,
            return_indices_ranked_by='self_confidence',
            min_examples_per_class=0,
            # n_classes= 10
        )

        # ordered_label_issues = find_label_issues_using_argmax_confusion_matrix(
        #     labels=labels,
        #     pred_probs = pred_probs,
        #     calibrate= True
        # )
        # print(ordered_label_issues)

        ordered_label_issues = ordered_label_issues.tolist()
        if len(ordered_label_issues) > 0:
            idxs = all_idxs[ordered_label_issues].astype(int)   # 默认是 float 型
            # print(temp)
            temp = idxs.tolist()
            drop_list += temp
        # print("return:",drop_list)
        return drop_list

    # 过滤函数将可能是以后的核心函数
    @torch.no_grad()
    def filter_data(self, dataloader):
        drop_list = []
        self.c_model.train()
        for gen_imgs, gen_labels, idxs in  dataloader:
            gen_imgs, gen_labels = gen_imgs.cuda(), gen_labels.cuda()
            logits_gen = self.c_model(gen_imgs)
            logits_gen = F.softmax(logits_gen, dim=1)
            preds, _ = torch.max(logits_gen, dim=1)      # 最大预测值
            print("pred mean:",preds.mean())
            # 调整
            confidence_mask = (preds <= 0.4)  # 0.5 0.4 0.35 获取置信度较高的 生成样本 ，日后动态调整阈值
            # 混乱度低的
            # uncertainty = -1.0 * torch.sum(logits_gen * torch.log(logits_gen + 1e-6), dim=1)  # 信息熵
            # uncertainty_mask = (uncertainty < 1.0)    # 2.0 4.0, 1.0  感觉没有意义, 经验证,无意义
            temp = idxs[confidence_mask]
            temp = temp.tolist()
            # temp= idxs[confidence_mask].tolist()
            drop_list += temp
        return drop_list

    # 过滤函数将可能是以后的核心函数, 使用课程伪标签
    @torch.no_grad()
    def filter_data_CPL(self, dataloader, n_sampler = 0):
        num_classes = 10
        thresh_warmup = False
        p_cutoff = 0.9

        selected_label = torch.ones( n_sampler, dtype=torch.long, ) * -1  # 初始化为 -1
        selected_label = selected_label.cuda()
        classwise_acc = torch.zeros((10,)).cuda()  # 初始化每个类的精度为 0

        drop_list = []
        self.c_model.train()
        for gen_imgs, gen_labels, idxs in  dataloader:
            gen_imgs, gen_labels = gen_imgs.cuda(), gen_labels.cuda()

            pseudo_counter = Counter(selected_label.tolist())   # pseudo_counter 存放了每类样本的数量
            if max(pseudo_counter.values()) < n_sampler:  # not all(5w) -1
                if thresh_warmup:
                    for i in range(num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            logits_gen = self.c_model(gen_imgs)
            logits_gen = F.softmax(logits_gen, dim=1)
            preds, max_idx = torch.max(logits_gen, dim=1)      # 最大预测值
            # torch.ge 比较大小
            mask = preds.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))).float()  # convex
            select = mask.ge(p_cutoff).long()
            if idxs[select == 1].nelement() != 0:      #.nelement() 统计元素数量
                selected_label[idxs[select == 1]] = max_idx[select == 1]    # 给高置信度样本赋予标签
            # print("pred mean:",preds.mean())
            # 调整
            # mask = get_mask()
            confidence_mask = (mask < 1.0)  # 0.5 0.4 0.35 获取置信度较高的 生成样本 ，日后动态调整阈值
            # 混乱度低的
            # uncertainty = -1.0 * torch.sum(logits_gen * torch.log(logits_gen + 1e-6), dim=1)  # 信息熵
            # uncertainty_mask = (uncertainty < 1.0)    # 2.0 4.0, 1.0  感觉没有意义, 经验证,无意义
            temp = idxs[confidence_mask]
            temp = temp.tolist()
            # temp= idxs[confidence_mask].tolist()
            drop_list += temp
        return drop_list

    @torch.no_grad()
    def evaluate_public(self, eval_loader):
        self.c_model.cuda()
        self.c_model.eval()  # ?
        # if use_ema == True:
        #     self.ema.apply_shadow()     # ?????
        if eval_loader is None:
            eval_loader = self.test_dataloader
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_logits = []
        for  x, y in eval_loader:
            # 输出 y的长度
            # print("y", len(y))
            x, y = x.cuda(), y.cuda()
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.c_model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            # y_logits.extend(torch.max(torch.softmax(logits, dim=-1),dim=-1))
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)       # sklearn.metrics 中的包， 简洁实现一次
        # self.print_fn(f"[client: {self.idx}] [accuracy: {top1}]")
        self.c_model.train()
        return top1

    @torch.no_grad()
    def evaluate(self, test_loader = None):
        self.c_model.cuda()
        self.c_model.eval()
        # if use_ema == True:
        #     self.ema.apply_shadow()     # ?????

        eval_loader = self.test_dataloader
        if test_loader != None:
            eval_loader = test_loader
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_logits = []
        for x, y in eval_loader:
        # for  x, y, _ in eval_loader:
            # 输出 y的长度
            # print("y", len(y))
            x, y = x.cuda(), y.cuda()
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.c_model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            # y_logits.extend(torch.max(torch.softmax(logits, dim=-1),dim=-1))
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)       # sklearn.metrics 中的包， 简洁实现一次
        # self.print_fn(f"[client: {self.idx}] [accuracy: {top1}]")
        self.c_model.train()
        return top1

    def train_1(self):
        for i, (imgs, labels) in enumerate(self.dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)  # pytorch tensor的操作
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            # labels = Variable(labels.type(LongTensor))
            labels = labels.type(LongTensor)

            # -----------------
            #  Train Generator
            # -----------------

            self.optimizer_G.zero_grad()

            z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
            # 生成的标签应该只在样本包含的标签中
            y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()  # ？rand（）产生0~1之间的数，squeeze去除1维度
            y_label_ = onehot[y_]  # 生成器  10 * 1 * 1， 此处可以不用数组，直接利用onehot()函数
            y_fill_ = fill[y_]  # 用于判别器 10 * 32 * 32  ，其中一个通道全为 1
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

            # Generate a batch of images
            # gen_imgs = generator(z_, gen_labels)
            gen_imgs = self.generator(z_, y_label_)

            # Loss measures generator's ability to fool the discriminator
            # validity = discriminator(gen_imgs, gen_labels)
            validity = self.discriminator(gen_imgs, y_fill_)
            g_loss = adversarial_loss(validity, valid)  # ？？？  判别的值和 1 作损失

            g_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Loss for real images
            # validity_real = discriminator(real_imgs, labels)
            labels_fill = fill[labels]
            # labels_fill = Variable(labels_fill) # error ,因为fill 在cpu中
            labels_fill = Variable(labels_fill.cuda())
            validity_real = self.discriminator(real_imgs, labels_fill)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            # validity_fake = discriminator(gen_imgs.detach(), gen_labels)    #？ detach 返回一个新的tensor，从当前计算图中分离下来。但是仍指向原变量的存放位置，不具有grad
            validity_fake = self.discriminator(gen_imgs.detach(), y_fill_)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            print(
                "[Client %d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (self.idx, i, len(self.dataloader), d_loss.item(), g_loss.item())
            )


    def save_model(self, save_name, save_path):     # 吸收
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        # self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        # self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):        # 吸收
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


# 内部使用
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
