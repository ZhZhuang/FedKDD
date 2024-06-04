#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])    # setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是存在的。

def args_parser():
    parser = argparse.ArgumentParser()
    conf_file = os.path.abspath("../conf/CIFAR_balance_conf.json")
    parser.add_argument('--conf', default=conf_file,
                        help='the config file for FedMD.'
                       )
    parser.add_argument('--use_pretrained_model', type=bool, default=False,
                        help="number of rounds of training")

    parser.add_argument('--gpu', type=float, default=None,
                        help='set gpu id')

    # 以下参数不修改
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")    # cifar or mnist
    parser.add_argument('--iid', type=bool, default=False,
                        help='Default set to non-IID. Set to True for IID.')

    args = parser.parse_args()
    data_set ='mnist' if 'MNIST' in args.conf else 'cifar'
    args.dataset = data_set
    # args.iid = False if 'imbalance' in args.conf else True
    args.iid = True
    print(args)
    return args

def args_parser_PSFL():
    parser = argparse.ArgumentParser()
    conf_file = os.path.abspath("../conf/CIFAR_balance_conf.json")
    parser.add_argument('--conf', default=conf_file,
                        help='the config file for FedMD.'
                       )
    parser.add_argument('--use_pretrained_model', type=bool, default=False,
                        help="number of rounds of training")

    parser.add_argument('--gpu', type=float, default=None,
                        help='set gpu id')

    # 以下参数不修改
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")    # cifar or mnist
    parser.add_argument('--iid', type=bool, default=False,
                        help='Default set to non-IID. Set to True for IID.')

    args = parser.parse_args()
    data_set ='mnist' if 'MNIST' in args.conf else 'cifar'
    args.dataset = data_set
    # args.iid = False if 'imbalance' in args.conf else True
    #args.iid = False
    args.iid = True
    print(args)
    return args

def str2bool(v):        # 吸收
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def args_parser_flexmatch():
#     parser = argparse.ArgumentParser(description='')
#
#     '''
#         Saving & loading of the model.
#         '''
#     parser.add_argument('--save_dir', type=str, default='./saved_models')
#     parser.add_argument('-sn', '--save_name', type=str, default='flexmatch')
#     parser.add_argument('--resume', action='store_true')
#     parser.add_argument('--load_path', type=str, default=None)
#     parser.add_argument('-o', '--overwrite', action='store_true')
#     parser.add_argument('--use_tensorboard', action='store_true',
#                         help='Use tensorboard to plot and save curves, otherwise save the curves locally.')
#
#     '''
#     Training Configuration of flexmatch
#     '''
#
#     parser.add_argument('--epoch', type=int, default=1)
#     parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
#                         help='total number of training iterations')
#     parser.add_argument('--num_eval_iter', type=int, default=5000,
#                         help='evaluation frequency')
#     parser.add_argument('-nl', '--num_labels', type=int, default=40)
#     parser.add_argument('-bsz', '--batch_size', type=int, default=64)
#     parser.add_argument('--uratio', type=int, default=7,
#                         help='the ratio of unlabeled data to labeld data in each mini-batch')
#     parser.add_argument('--eval_batch_size', type=int, default=1024,
#                         help='batch size of evaluation data loader (it does not affect the accuracy)')
#
#     parser.add_argument('--hard_label', type=str2bool, default=True)
#     parser.add_argument('--T', type=float, default=0.5)
#     parser.add_argument('--p_cutoff', type=float, default=0.95)
#     parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
#     parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
#     parser.add_argument('--use_DA', type=str2bool, default=False)
#     parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=True)
#
#     '''
#     Optimizer configurations
#     '''
#     parser.add_argument('--optim', type=str, default='SGD')
#     parser.add_argument('--lr', type=float, default=3e-2)
#     parser.add_argument('--momentum', type=float, default=0.9)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)
#     parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
#     parser.add_argument('--clip', type=float, default=0)
#     '''
#     Backbone Net Configurations
#     '''
#     parser.add_argument('--net', type=str, default='WideResNet')
#     parser.add_argument('--net_from_name', type=str2bool, default=False)
#     parser.add_argument('--depth', type=int, default=28)
#     parser.add_argument('--widen_factor', type=int, default=2)
#     parser.add_argument('--leaky_slope', type=float, default=0.1)
#     parser.add_argument('--dropout', type=float, default=0.0)
#
#     '''
#     Data Configurations
#     '''
#
#     parser.add_argument('--data_dir', type=str, default='./data')
#     parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
#     parser.add_argument('--train_sampler', type=str, default='RandomSampler')
#     parser.add_argument('-nc', '--num_classes', type=int, default=10)
#     parser.add_argument('--num_workers', type=int, default=1)
#
#     '''
#     multi-GPUs & Distrbitued Training
#     '''
#
#     ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
#     parser.add_argument('--world-size', default=1, type=int,
#                         help='number of nodes for distributed training')
#     parser.add_argument('--rank', default=0, type=int,
#                         help='**node rank** for distributed training')
#     parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:10601', type=str,
#                         help='url used to set up distributed training')
#     parser.add_argument('--dist-backend', default='nccl', type=str,
#                         help='distributed backend')
#     parser.add_argument('--seed', default=1, type=int,
#                         help='seed for initializing training. ')
#     parser.add_argument('--gpu', default=None, type=int,
#                         help='GPU id to use.')
#     parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
#                         help='Use multi-processing distributed training to launch '
#                              'N processes per node, which has N GPUs. This is the '
#                              'fastest way to use PyTorch for either single node or '
#                              'multi node data parallel training')
#
#     # config file
#     # parser.add_argument('--c', type=str, default='')
#
#     # parser.add_argument('--c', type=str, default='../conf/pimodel/pimodel_cifar10_40_0.yaml')
#     # parser.add_argument('--c', type=str, default='./config/pimodel/pimodel_cifar10_4000_0.yaml')
#     parser.add_argument('--c', type=str, default='./config/flexmatch/flexmatch_cifar10_4000_0.yaml')
#     args = parser.parse_args()
#     over_write_args_from_file(args, args.c)
#     return args


def args_parser_flexmatch(file = None):
    if file != None:
        file = os.path.join("config",file)
    else:
        file = './config/flexmatch/flexmatch_cifar10_4000_0.yaml'
    parser = argparse.ArgumentParser(description='')
    '''
        Saving & loading of the model.
        '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='flexmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of flexmatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--use_DA', type=str2bool, default=False)
    parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=True)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:10601', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--is_fullySupervised', default=False, type=bool,
                        help='is fullySupervised')

    # config file
    # parser.add_argument('--c', type=str, default='')

    # parser.add_argument('--c', type=str, default='../conf/pimodel/pimodel_cifar10_40_0.yaml')
    # parser.add_argument('--c', type=str, default='./config/pimodel/pimodel_cifar10_4000_0.yaml')
    parser.add_argument('--c', type=str, default=file)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args

def args_parser_pimodel(file = None):
    if file != None:
        file = os.path.join("config",file)
    else:
        file = 'config/pimodel/pimodel_cifar10_4000_0.yaml'
    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='pimodel')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of PiModel
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=1000,
                        help='evaluation frequency')
    parser.add_argument('--unsup_warmup_pos', type=float, default=0.4,
                        help='Relative position at which constraint loss warmup ends.')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--ema_m', type=float, default=0.999)
    parser.add_argument('--ulb_loss_ratio', type=float, default=10.0)
    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10002', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--is_fullySupervised', default=False, type=bool,
                        help='is fullySupervised')

    # config file
    # parser.add_argument('--c', type=str, default='')
    parser.add_argument('--c', type=str, default=file)

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args

def args_parser_fixmatch(file = None):
    if file != None:
        file = os.path.join("config",file)
    else:
        file = 'config/fixmatch/fixmatch_cifar10_4000_0.yaml'
    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--is_fullySupervised', default=False, type=bool,
                        help='is fullySupervised')
    # config file
    # parser.add_argument('--c', type=str, default='')

    parser.add_argument('--c', type=str, default=file)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args

def args_parser_fullysupervised(file = None):
    if file != None:
        file = os.path.join("config",file)
    else:
        file = './config/fullysupervised/fullysupervised_cifar10__0.yaml'

    parser = argparse.ArgumentParser(description='')
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='fullysupervised')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of FullySupervised Learning
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=1,
                        # to keep simple, we set uratio = 1, but we don't use unlabeled data at all
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--ema_m', type=float, default=0.999)
    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10002', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--is_fullySupervised', default=False, type=bool,
                        help='is fullySupervised')
    # config file
    parser.add_argument('--c', type=str, default=file)

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args

if __name__ == '__main__':
    # args = args_parser()
    args = args_parser_flexmatch()
    print('finish')