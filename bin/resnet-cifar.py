# -*- coding: utf-8 -*-
import os
import sys
import logging

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from skeleton.resnet import ResNet
from skeleton.datasets import Cifar224


def main(args):
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    batch_size = args.batch
    train_loader, valid_loader, test_loader, data_shape = Cifar224.loader(batch_size, args.num_class)
    _ = test_loader

    model = ResNet(args.num_class)

    # Print layer shapes.
    with torch.no_grad():
        model(torch.Tensor(*data_shape[0]), verbose=True)

    # Enable data parallelism.
    model = nn.DataParallel(model)
    model.to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4 * batch_size, momentum=0.9)

    for epoch in range(args.epoch):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.info('[train] [epoch:%04d/%04d] [step:%04d/%04d] loss: %.5f',
                         epoch, args.epoch, batch_idx + 1, len(train_loader), float(loss))

        with torch.no_grad():
            losses = []

            total = 0
            correct = 0

            for inputs, targets in valid_loader:
                targets = targets.to(device)

                outputs = model(inputs)

                # loss
                loss = float(F.cross_entropy(outputs, targets))
                losses.append(loss)

                # accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            loss = np.average(losses)
            accuracy = correct / total

            logging.info('[vaild] [epoch:%04d/%04d]                  loss: %.5f, accuracy: %.1f%%',
                         epoch, args.epoch, loss, accuracy * 100)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num-class', type=int, default=10, help='10 or 100')
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=25)

    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    main(parsed_args)
