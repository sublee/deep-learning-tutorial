# -*- coding: utf-8 -*-
from torch import nn

from skeleton.nn.modules import Flatten, GlobalPool, IOModule


__all__ = ['ResNet']


class ResNet(IOModule):

    def __init__(self, num_classes=10):
        super().__init__()

        # the first simple convolution layer
        self.conv1 = nn.Sequential(
            # The input size should be (n, 3, 224, 224).
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # resedual blocks
        self.conv2 = ResedualBlock(in_channels=64, mid_channels=64, out_channels=256, stride=1, repeat=3)
        self.conv3 = ResedualBlock(in_channels=256, mid_channels=128, out_channels=512, stride=2, repeat=4)
        self.conv4 = ResedualBlock(in_channels=512, mid_channels=256, out_channels=1024, stride=2, repeat=6)
        self.conv5 = ResedualBlock(in_channels=1024, mid_channels=512, out_channels=2048, stride=2, repeat=3)

        # final layers
        self.gap = GlobalPool()
        self.fc = nn.Sequential(Flatten(), nn.Linear(in_features=2048, out_features=num_classes))

    def forward(self, x, verbose=False):  # pylint: disable=arguments-differ
        if verbose:
            print('%20s shape after %s' % (list(x.size()), 'inputs'))
        for name, layer in self.named_children():
            x = layer.forward(x)
            if verbose:
                print('%20s shape after %s' % (list(x.size()), name))
        return x


class ResedualBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int, repeat: int):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(repeat):
            block = nn.ModuleDict({
                'body': nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(),

                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=(stride if i == 0 else 1), padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(),

                    nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                ),
                'relu': nn.ReLU(),
            })

            if i == 0 and in_channels != out_channels:
                block['down_sample'] = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1)

            self.blocks.append(block)

            in_channels = out_channels

    def forward(self, x):  # pylint: disable=arguments-differ
        for i, block in enumerate(self.blocks):
            x_keep = x

            x = block['body'](x)

            if i == 0 and 'down_sample' in block:
                x_keep = block['down_sample'](x_keep)
            x += x_keep

            x = block['relu'](x)
        return x
