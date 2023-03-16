from .transforms import make_transforms
from . import samplers
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config import cfg
import cv2
cv2.setNumThreads(1)

# torch.multiprocessing.set_sharing_strategy('file_system')


def _dataset_factory(is_train):
    if is_train:
        path = cfg.train_dataset_path
        args = cfg.train_dataset
    else:
        path = cfg.test_dataset_path
        args = cfg.test_dataset
    module = path[:-3].replace('/', '.')
    dataset = imp.load_source(module, path).Dataset(**args)
    return dataset


def make_dataset(cfg, transforms, is_train=True):
    dataset = _dataset_factory(is_train)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler':
        sampler = samplers.FrameSampler(dataset)
        return sampler
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
    else:
        batch_sampler = cfg.test.batch_sampler

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    cv2.setNumThreads(1)
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = True
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    transforms = make_transforms(cfg, is_train)
    dataset = make_dataset(cfg, transforms, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn)

    return data_loader
