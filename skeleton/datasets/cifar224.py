# -*- coding: utf-8 -*-
import logging

import skorch
import numpy as np
import torch
import torchvision as tv

LOGGER = logging.getLogger(__name__)

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


class Cifar224:
    @staticmethod
    def datasets(num_classes=10, cv_ratio=0.2, cv_seed=None, root='./data'):
        assert num_classes in [10, 100]

        dataset = tv.datasets.CIFAR10 if num_classes == 10 else tv.datasets.CIFAR100
        data_shape = [(3, 224, 224), (1,)]

        transform_train = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        transform_valid = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        train_set = dataset(root=root, train=True, download=True, transform=transform_train)
        test_set = dataset(root=root, train=False, download=True, transform=transform_valid)

        spliter = skorch.dataset.CVSplit(cv=cv_ratio, stratified=True, random_state=cv_seed)
        train_set, valid_set = spliter(train_set, y=np.array(train_set.train_labels, dtype=np.int8))

        return train_set, valid_set, test_set, data_shape

    @staticmethod
    def loader(batch_size, num_classes=10, cv_ratio=0.2, cv_seed=None, root='./data', num_workers=8):
        train_set, valid_set, test_set, data_shape = Cifar224.datasets(num_classes, cv_ratio, cv_seed, root)

        data_shape = [(batch_size,) + shape for shape in data_shape]

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

        return train_loader, valid_loader, test_loader, data_shape
