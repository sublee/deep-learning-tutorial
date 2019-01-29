# -*- coding: utf-8 -*-
import math

from torch import nn

from skeleton.nn.modules import Flatten, GlobalPool, IOModule


__all__ = ['ResNet50']


class ResNet50(IOModule):
    """ResNet-50 in `Deep Residual Learning for Image Recognition by Microsoft
    Research <https://arxiv.org/abs/1512.03385>`_.
    """

    def __init__(self, num_classes=10, input_size=224):
        super().__init__()

        # the first simple convolution layer
        layer1 = [nn.BatchNorm2d(64), nn.ReLU()]

        assert input_size in [224, 32]

        if input_size == 224:
            layer1.insert(0, nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False))
            layer1.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        elif input_size == 32:
            layer1.insert(0, nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))

        # The input size should be (n, 3, 224, 224)wor (n, 3, 32, 32).
        self.layer1 = nn.Sequential(*layer1)

        # residual blocks
        self.layer2 = ResidualBlock(in_channels=64, mid_channels=64, out_channels=256, stride=1, repeat=3)
        self.layer3 = ResidualBlock(in_channels=256, mid_channels=128, out_channels=512, stride=2, repeat=4)
        self.layer4 = ResidualBlock(in_channels=512, mid_channels=256, out_channels=1024, stride=2, repeat=6)
        self.layer5 = ResidualBlock(in_channels=1024, mid_channels=512, out_channels=2048, stride=2, repeat=3)

        # final layers
        self.gap = GlobalPool()
        self.fc = nn.Sequential(Flatten(), nn.Linear(in_features=2048, out_features=num_classes))

        # init weight (He initialization)
        self.apply(self._init_weight)

    @staticmethod
    def _init_weight(m):
        """Initializes weight with the He initialization."""
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            return

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            return

    def forward(self, x, verbose=False):  # pylint: disable=arguments-differ
        """
        :param verbose: If ``True``, prints shape of input after each child layers.
        """
        if verbose:
            print('%20s shape after %s' % (list(x.size()), 'inputs'))
        for name, layer in self.named_children():
            x = layer.forward(x)
            if verbose:
                print('%20s shape after %s' % (list(x.size()), name))
        return x


class ResidualBlock(nn.Sequential):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int, repeat: int):
        blocks = []
        blocks.append(Bottleneck(in_channels, mid_channels, out_channels, stride))

        for _ in range(1, repeat):
            blocks.append(Bottleneck(out_channels, mid_channels, out_channels, 1))

        super().__init__(*blocks)


class Bottleneck(nn.Module):
    """A bottleneck building block in ResNet. It is used for ResNet-50/101/152.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int):
        super().__init__()

        # layer 1: in->mid
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )

        # layer 2: mid->mid with stride
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )

        # layer 3: mid->out with a shortcut connection
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            downsample = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        else:
            # nothing to downsample
            downsample = []
        self.downsample = nn.Sequential(*downsample)

        self.layer3_relu = nn.ReLU()

    def forward(self, x):  # pylint: disable=arguments-differ
        x_keep = x

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x += self.downsample(x_keep)
        x = self.layer3_relu(x)

        return x
