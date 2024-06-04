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

    def set_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def train_net_fedprox(self , config, global_model = None):
        mu = config.mu
        device = config.device
        epochs = config.local_epoch

        train_dataloader = self.dataloader
        optimizer = self.optimizer_C
        self.c_model = self.c_model.to(device)
        # writer = SummaryWriter()
        global_weight_collector = list(global_model.to(device).parameters())

        for epoch in range(epochs):
            epoch_loss_collector = []
            # for tmp in train_dataloader:
            for batch_idx, (x, target, _) in enumerate(train_dataloader):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = self.c_model(x)
                loss = self.criterion(out, target)
                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.c_model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())



    @torch.no_grad()
    def evaluate_public(self, eval_loader, model = None):

        # self.c_model.eval()  # ?
        Model = self.c_model
        if model != None :
            Model = model
        Model = Model.cuda()
        Model.eval()
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
            logits = Model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            # y_logits.extend(torch.max(torch.softmax(logits, dim=-1),dim=-1))
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)       # sklearn.metrics 中的包， 简洁实现一次
        # self.print_fn(f"[client: {self.idx}] [accuracy: {top1}]")
        Model.train()
        return top1

    @torch.no_grad()
    def evaluate(self):
        self.c_model.cuda()
        self.c_model.eval()  # ?
        # if use_ema == True:
        #     self.ema.apply_shadow()     # ?????
        eval_loader = self.test_dataloader
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_logits = []
        for  x, y, _ in eval_loader:
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
