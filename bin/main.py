# -*- coding: utf-8 -*-
from bisect import bisect_right
from datetime import datetime
import logging
import os
import sys
import time

from tensorboardX import SummaryWriter
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from skeleton.resnet import ResNet50
from skeleton.datasets import Cifar, Cifar224, Imagenet
from skeleton.utils import init_process_group, Noop, TensorBoardWriter


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

    rank, world_size = init_process_group()

    # Use only 1 GPU.
    device = torch.device('cuda', args.local_rank + args.from_rank)
    torch.cuda.set_device(device)

    # Data loaders.
    (train_set, valid_set, data_shape), num_classes, num_epochs, lr_warmup, lr_milestones = load_env(args.batch, args.data)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=1, pin_memory=True, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, num_workers=1, pin_memory=True, drop_last=False)

    # Init the model.
    input_size = data_shape[0][2]
    model = ResNet50(num_classes, input_size)
    model.to(device=device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)

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
    initial_lr = 0.0004 * args.batch * world_size
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)

    # LR scheduling.
    def lr_schedule(epoch):
        if epoch < lr_warmup:
            # gradual warmup
            inv_world_size = (1 / world_size)
            return inv_world_size + ((1 - inv_world_size) / lr_warmup * epoch)

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
            step_t = time.time()

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

        # record time per epoch
        tb(epoch + 1).scalar('time-per/epoch', time.time() - epoch_t)

    def valid(epoch):
        model.eval()

        losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
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
