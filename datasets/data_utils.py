import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
import numpy as np
import json
import os

from datasets.DistributedProxySampler import DistributedProxySampler


def split_ssl_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, num_classes, index) #
    # sorted 生成一个排好序的列表, 下面用不用sorted感觉意义不大
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        # 下面是numpy的 索引操作，见过好几次了
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]

# 会产生一个 sampled_label_idx.npy 文件，保存有标签的样本 下标
def sample_labeled_data(args, data, target,
                        num_labels, num_classes,
                        index=None, name=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # 每个类的标签数都是相同的，否这报错 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("num_labels:",num_labels, " num_classes:",num_classes)
    # assert num_labels % num_classes == 0    # ! 非独立同分布情况需要注释掉
    if not index is None:   # 根据 index直接生成样本
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    # 将有将有标签的数据的 idx 保存在 sampled_label_idx.npy 中
    dump_path = os.path.join(args.save_dir, args.save_name, 'sampled_label_idx.npy')

    # 注释掉 兼容超算
    # if os.path.exists(dump_path):   # 路径存在的话直接直接读取，因此修改样本数的时候会出现问题
    #     lb_idx = np.load(dump_path)
    #     lb_data = data[lb_idx]
    #     lbs = target[lb_idx]
    #     return lb_data, lbs, lb_idx

    # samples_per_class = int(num_labels / num_classes)   # 改
    samples_per_class = int(num_labels)       # 改变 num_labels 为每类的样本数

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]  # 获取 类为 c 的全部索引
        if len(idx)>samples_per_class:     # 判断
            print("idx", len(idx), "per_class", samples_per_class)
            idx = np.random.choice(idx, samples_per_class, False)   # noniid情况，
            lb_idx.extend(idx)
            lb_data.extend(data[idx])
            lbs.extend(target[idx])


    np.save(dump_path, np.array(lb_idx))    # 将有将有标签的数据的 idx 保存在 sampled_label_idx.npy 中

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__
                               if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)   # 直接返回 类名， 未实例化
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_data_loader(dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler=None,
                    replacement=True,
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):  # here
            num_samples = batch_size * num_iters * num_replicas     # 决定了样本数量
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == 'RandomSampler':
            # data_sampler = data_sampler(dset, replacement, num_samples, generator)  # 实例化
            data_sampler = data_sampler(dset, replacement, num_samples )  # 兼容超算
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
