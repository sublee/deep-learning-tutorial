# -*- coding: utf-8 -*-
from bisect import bisect_right
from datetime import datetime
import logging
import os
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from skeleton.resnet import ResNet50
from skeleton.datasets import Cifar, Cifar224, Imagenet
from skeleton.utils import Noop, TensorBoardWriter


assert torch.cuda.is_available()


def correct_total(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return (correct, total)


def find_lr(optimizer):
    for param_group in optimizer.param_groups:
        try:
            return param_group['lr']
        except KeyError:
            pass


def load_env(batch_size, dataset_name) -> (('train_set', 'valid_set', 'data_shape'), 'num_classes', 'num_epochs', 'lr_warmup', 'lr_milestones'):
    if dataset_name == 'cifar10':
        return (Cifar.sets(batch_size, 10),
                10,
                300,
                15,
                [100, 150, 200])

    if dataset_name == 'cifar100':
        return (Cifar.sets(batch_size, 100),
                100,
                300,
                15,
                [100, 150, 200])

    if dataset_name == 'cifar10-224':
        return (Cifar224.sets(batch_size, 10),
                10,
                300,
                15,
                [100, 150, 200])

    if dataset_name == 'cifar100-224':
        return (Cifar224.sets(batch_size, 100),
                100,
                300,
                15,
                [100, 150, 200])

    if dataset_name == 'imagenet':
        return (Imagenet.sets(batch_size),
                1000,
                100,
                5,
                [30, 60, 80])


def main(args):
    logging.info('args: %s', args)

    # Init cluster on a launcher.
    launcher = args.launcher

    if launcher == 'auto':
        # Detect launcher.
        if os.getenv('OMPI_UNIVERSE_SIZE'):
            launcher = 'horovod'
        else:
            launcher = 'pytorch'

    assert launcher in ['pytorch', 'horovod']

    if launcher == 'pytorch':
        from skeleton.utils import init_process_group
        rank, world_size = init_process_group()
        local_rank = args.local_rank

    elif launcher == 'horovod':
        import horovod.torch as hvd
        hvd.init()
        rank, world_size = hvd.rank(), hvd.size()
        local_rank = hvd.local_rank()

    # Use only 1 GPU.
    device = torch.device('cuda', local_rank + args.from_rank)
    torch.cuda.set_device(device)

    # Data loaders.
    (train_set, valid_set, data_shape), num_classes, num_epochs, lr_warmup, lr_milestones = load_env(args.batch, args.data)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=10, pin_memory=True, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, num_workers=10, pin_memory=True, drop_last=False)

    # Init the model.
    dtype = torch.float16
    input_size = data_shape[0][2]

    model = ResNet50(num_classes, input_size)
    model.to(device, dtype)

    # Convert input batches to float16 in the pin memory thread of DataLoader.
    if dtype is torch.float16:
        from torch import Tensor
        from torch.utils.data import dataloader

        def to_half(f):
            def wrapped(*args, **kwargs):
                x = f(*args, **kwargs)
                if isinstance(x, Tensor) and x.is_floating_point():
                    return x.half()
                return x
            return wrapped

        dataloader.pin_memory_batch = to_half(dataloader.pin_memory_batch)

    # Integrate with TensorBoard.
    if rank == 0:
        if args.run:
            run_name = '{:%m-%d/%H:%M} {}'.format(datetime.now(), args.run)
            tb_path = os.path.join(args.run_dir, run_name)
        else:
            tb_path = None
        tb = TensorBoardWriter(len(train_loader), tb_path)
    else:
        tb = Noop()

    # Optimization strategy.
    total_batch_size = args.batch * world_size
    initial_lr = 0.1 / 256 * total_batch_size
    if dtype is torch.float16:
        initial_lr /= 2
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)

    # Distributed learning.
    if launcher == 'pytorch':
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)

    elif launcher == 'horovod':
        optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters())
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # LR scheduling.
    def lr_schedule(epoch):
        if epoch < lr_warmup:
            # gradual warmup
            return (256 / total_batch_size) + (1 - 256 / total_batch_size) / lr_warmup * epoch

        # multi-step LR schedule (1/10 at each LR milestone)
        return 0.1 ** bisect_right(lr_milestones, epoch - lr_warmup)

    scheduler = LambdaLR(optimizer, lr_schedule)

    # -------------------------------------------------------------------------

    def step(epoch):
        train_sampler.set_epoch(epoch)
        scheduler.step(epoch)
        tb(epoch).scalar('lr', find_lr(optimizer))

    def train(epoch):
        model.train()

        epoch_t = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            # NOTE: gradual warmup should be at each iteration instead of epoch.
            if epoch < lr_warmup:
                scheduler.step(epoch + (i / len(train_loader)))

            step_t = time.time()

            inputs = inputs.to(device, dtype)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tb_add = tb(epoch, i)
            if tb_add:
                tb_add.scalar('time-per/step', time.time() - step_t)

                correct, total = correct_total(outputs, targets)
                accuracy = correct / total
                tb_add.scalar('accuracy/train', accuracy)

                tb_add.scalar('loss/train', float(loss))

        # record for every epoch
        time_per_epoch = time.time() - epoch_t
        epoch_batch_size = len(train_loader) * total_batch_size
        tb(epoch + 1).scalar('time-per/epoch', time_per_epoch)
        tb(epoch + 1).scalar('batch-per-second', epoch_batch_size / time_per_epoch)

    def valid(epoch):
        model.eval()

        losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(device, dtype)
                targets = targets.to(device)

                outputs = model(inputs)

                # loss
                loss = float(F.cross_entropy(outputs, targets))
                losses.append(loss)

                # accuracy
                correct_, total_ = correct_total(outputs, targets)
                correct += correct_
                total += total_

        accuracy = correct / total
        tb(epoch + 1).scalar('accuracy/valid', accuracy)

        loss = np.average(losses)
        tb(epoch + 1).scalar('loss/valid', float(loss))

    for epoch in range(num_epochs):
        step(epoch)
        train(epoch)
        valid(epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--launcher', type=str, default='auto', help='[auto]|pytorch|horovod')

    parser.add_argument('-d', '--data', type=str, default='cifar10')
    parser.add_argument('-b', '--batch', type=int, default=128)

    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--run-dir', type=str, default='runs')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--from-rank', type=int, default=0)
    parsed_args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    main(parsed_args)
