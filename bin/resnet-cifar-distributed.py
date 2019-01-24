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
from skeleton.datasets import Cifar224


assert torch.cuda.is_available()


def correct_total(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return (correct, total)


def init_process_group():
    while True:
        try:
            dist.init_process_group('nccl')
        except (RuntimeError, ValueError):
            # RuntimeError: Connection timed out
            time.sleep(5)
            continue
        else:
            break

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logging.info('process group: rank-%d among %d processes' % (rank, world_size))

    return rank, world_size


class Noop:
    def __init__(self, *args, **kwargs):
        pass
    def noop(self, *args, **kwargs):
        return self
    __call__ = __getattr__ = __getitem__ = noop


class SummaryPrinter:
    def add_scalar(self, name, value, step):
        logging.info('%s: %.5f [step:%d]' % (name, value, step))


def gen_global_step(epoch, batch_idx=0, num_batches=0):
    """Generates a global step by the current epoch, batch index, and the
    number of batches.

    Returns:
        (current global step, whether if the global step changed at this batch index)

    """
    def f(batch_idx, steps_per_epoch=1000):
        return epoch * steps_per_epoch + (int(batch_idx / num_batches * steps_per_epoch) if num_batches else 0)

    current = f(batch_idx)
    prev = f(batch_idx - 1)

    return current, (current != prev)


def main(args):
    logging.info('args: %s', args)

    rank, world_size = init_process_group()

    device = torch.device('cuda', args.local_rank + args.from_rank)
    torch.cuda.set_device(device)

    # Data loaders.
    train_set, valid_set, _, data_shape = Cifar224.datasets(args.num_class, cv_seed=0)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=1, pin_memory=True, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, num_workers=1, pin_memory=True, drop_last=False)

    # Init the model.
    model = ResNet50(args.num_class)
    model.to(device=device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)

    # Integrate with TensorBoard.
    if rank == 0:
        if args.run:
            run_name = '{:%m-%d/%H:%M} {}'.format(datetime.now(), args.run)
            tb = SummaryWriter(os.path.join(args.run_dir, run_name))
        else:
            tb = SummaryPrinter()
    else:
        tb = Noop()

    # Optimization strategy.
    initial_lr = 0.0004 * args.batch * world_size
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)

    def lr_schedule(epoch):
        if epoch < 5:
            # gradual warmup
            inv_world_size = (1 / world_size)
            return inv_world_size + ((1 - inv_world_size) / 5 * epoch)

        # multi-step LR schedule (1/10 at 30th, 60th, and 80th)
        milestones = [30, 60, 80]
        gamma = 0.1
        return gamma ** bisect_right(milestones, epoch)

    scheduler = LambdaLR(optimizer, lr_schedule)

    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)

        # adjust LR by epoch
        scheduler.step()

        # log LR
        global_step, _ = gen_global_step(epoch)
        for param_group in optimizer.param_groups:
            try:
                lr = param_group['lr']
            except KeyError:
                continue
            tb.add_scalar('lr', lr, global_step)
            break

        # train
        epoch_t = step_t = time.time()

        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            global_step, global_step_changed = gen_global_step(epoch, batch_idx, len(train_loader))

            targets = targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            correct, total = correct_total(outputs, targets)
            accuracy = correct / total

            logging.info('[train] [epoch:%04d/%04d] [step:%04d/%04d] loss: %.5f',
                         epoch + 1, args.epoch, batch_idx + 1, len(train_loader), float(loss))

            next_step_t = time.time()

            if global_step_changed:
                tb.add_scalar('loss/train', float(loss), global_step)
                tb.add_scalar('accuracy/train', accuracy, global_step)
                tb.add_scalar('time-per/step', next_step_t - step_t, global_step)

            step_t = next_step_t

        global_step, _ = gen_global_step(epoch + 1)

        # record time per epoch
        tb.add_scalar('time-per/epoch', time.time() - epoch_t, global_step)

        # validate
        model.eval()
        with torch.no_grad():
            losses = []

            correct = 0
            total = 0

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

            loss = np.average(losses)
            accuracy = correct / total

            logging.info('[vaild] [epoch:%04d/%04d]                  loss: %.5f, accuracy: %.1f%%',
                         epoch + 1, args.epoch, loss, accuracy * 100)

            tb.add_scalar('loss/valid', float(loss), global_step)
            tb.add_scalar('accuracy/valid', accuracy, global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=25)

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
