import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision

# Implements projections for residual connection dimensionality addition
# Option C from Paper https://arxiv.org/pdf/1512.03385.pdf

class ResNetConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, k_size, batch_norm, use_act) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=k_size, stride=1, padding=k_size//2
        )
        self.norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_ch)

        self.use_act = use_act
        if self.use_act:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm:
            x = self.batchnorm2d(x)
        if self.use_act:
            x = self.relu(x)
        return x

class ResNet18Initblock(nn.Module):

    def __init__(self, block, batch_norm = True) -> None:
        super().__init__()
        # special 7x7 convolution
        self.block1 = ResNetConvBlock(block[0], block[1], 7, batch_norm, True)
        self.block2 = ResNetConvBlock(block[1], block[2], 3, batch_norm, True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class ResNet18Block(nn.Module):

    def __init__(self, block, k_size=3, batch_norm = True) -> None:
        super().__init__()

        self.block1 = ResNetConvBlock(block[0], block[1], k_size, batch_norm, True)
        self.block2 = ResNetConvBlock(block[1], block[2], k_size, batch_norm, False)
        self.relu = nn.ReLU()
        # 1x1 conv for the residual, Implements Projections
        self.residual_conv = nn.Conv2d(block[0], block[2], kernel_size=1)

    def forward(self, x):
        residual = x
        out = x.clone()
        out = self.block1(out)
        out = self.block2(out)
        # add residual
        residual = self.residual_conv(residual)
        out = out + residual
        out = self.relu(out)
        return out


class Resnet18Upblock(nn.Module):

    def __init__(self, block, batch_norm = True, cut_channels_on_upsample = False) -> None:
        super().__init__()
        # to be called seperately it needs to have the same self.upsample and self.conv_block names

        self.upsample =  nn.ConvTranspose2d(
            in_channels=block[0],
            out_channels=block[0] // 2 if cut_channels_on_upsample else block[0],
            kernel_size=2,
            stride=2,
        )
        self.conv_block = ResNet18Block(block=block, batch_norm=batch_norm)

    def forward(self, x, temporal_state):
        x = self.upsample(x)
        x = torch.cat([x, temporal_state], dim=1)
        x = self.conv_block(x)
        return x

class Resnet18Downblock(nn.Module):

    def __init__(self, block, use_pooling, batch_norm = True) -> None:
        super().__init__()

        if use_pooling:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Conv2d(
                in_channels=block[0],
                out_channels=block[0],
                kernel_size=2,
                stride=2,
            )
        self.resnet18block = ResNet18Block(block=block, batch_norm=batch_norm)

    def forward(self, x):
        x = self.downsample(x)
        x = self.resnet18block(x)
        return x



