import torch
import torch.nn as nn
import torch.nn.functional as F

# following this paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705122007572
# convnext block from https://github.com/1914669687/ConvUNeXt/blob/master/src/ConvUNeXt.py

class ConvNextBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = self.act2(residual + x)

        return x

class SimpleConvBlock(nn.Module):
    """
    """

    def __init__(self, in_ch, out_ch, k_size, batch_norm, use_act) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=k_size, stride=1, padding=k_size//2
        )
        self.norm = batch_norm
        self.batchnorm = nn.BatchNorm2d(out_ch)

        self.use_act = use_act
        if self.use_act:
            self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm:
            x = self.batchnorm(x)
        if self.use_act:
            x = self.gelu(x)
        return x

class ConvNextLikeInitblock(nn.Module):

    def __init__(self, block, batch_norm = True) -> None:
        super().__init__()

        self.block1 = SimpleConvBlock(block[0], block[2], 7, batch_norm, True)
        self.convnextblock = ConvNextBlock(block[2])

    def forward(self, x):
        x = self.block1(x)
        x = self.convnextblock(x)
        return x

class ConvNextUpblock(nn.Module):

    def __init__(self, block, batch_norm = True, concat_hidden = True) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(block[0])
        if concat_hidden:
            C = block[0] // 2
        else:
            C = block[0]
        self.up = nn.ConvTranspose2d(block[0], block[0] // 2, kernel_size=2, stride=2)
        self.gate = nn.Linear(C, 3 * C)
        self.linear1 = nn.Linear(C, C)
        self.linear2 = nn.Linear(C, C)
        self.conv1x1 = nn.Conv2d(block[0], block[2], kernel_size=1)
        self.conv = ConvNextBlock(block[2])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)

        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        #attention
        B, C, H, W = x1.shape
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        gate = self.gate(x1).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        x2 = torch.sigmoid(self.linear1(g1 + x2)) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        x2 = self.linear2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x

class ConvNextDownblock(nn.Module):

    def __init__(self, block, use_pooling, batch_norm = True) -> None:
        super().__init__()

        if use_pooling:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Conv2d(
                in_channels=block[0],
                out_channels=block[2],
                kernel_size=2,
                stride=2,
            )
        self.convnextblock = ConvNextBlock(block[2])

    def forward(self, x):
        x = self.downsample(x)
        x = self.convnextblock(x)
        return x

