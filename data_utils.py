import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(local_rank, hp):
    # if local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((hp.data.image_size, hp.data.image_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((hp.data.image_size, hp.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if hp.data.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=hp.data.path,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=hp.data.path,
                                   train=False,
                                   download=True,
                                   transform=transform_test) if local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root=hp.data.path,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=hp.data.path,
                                    train=False,
                                    download=True,
                                    transform=transform_test) if local_rank in [-1, 0] else None
    # if local_rank == 0:
    #     torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if local_rank == 0 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=hp.train.batch,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=hp.train.valid_batch,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader