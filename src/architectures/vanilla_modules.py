from typing import Union
import torch
import torch.nn as nn


class ConvWithNorm(nn.Module):
    """Convolution + BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        padding: int = 0,
        stride: int = 1,
        batch_norm: bool = True,
    ) -> None:
        super(ConvWithNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.batchnorm2d(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Sequential):
    """Convolutional block."""

    def __init__(
        self,
        channels: list[int],
        batch_norm: bool = True,
    ) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        """
        super(ConvBlock, self).__init__(
            *[
                ConvWithNorm(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    batch_norm=batch_norm,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )


class UpsampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        channels: list[int],
        batch_norm: bool = True,
        cut_channels_on_upsample: bool = False,
    ) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        """
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=channels[0],
            out_channels=channels[0] // 2 if cut_channels_on_upsample else channels[0],
            kernel_size=2,
            stride=2,
        )
        self.conv_block = ConvBlock(channels, batch_norm)

    def forward(self, x, temporal_state):
        x = self.upsample(x)
        x = torch.cat([x, temporal_state], dim=1)
        x = self.conv_block(x)
        return x


class DownsampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        channels: list[int],
        use_pooling: bool = False,
        batch_norm: bool = True,
    ) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        """
        super(DownsampleBlock, self).__init__()
        if use_pooling:
            self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.downsampling = nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=2,
                stride=2,
            )
        self.conv_block = ConvBlock(channels, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsampling(x)
        x = self.conv_block(x)
        return x


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.modules.conv._ConvNd):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def center_pad(
    x: torch.Tensor,
    size: tuple[int, ...],
    offset: tuple[int, ...] = 0,
    mode: str = "constant",
    value: float = 0,
) -> torch.Tensor:
    """Center pad (or crop) nd-Tensor.
    Args:
        x (torch.Tensor): The Tensor to pad. The last `len(size)` dimensions will be padded.
        size (tuple[int, ...]): The desired pad size.
        offset (tuple[int, ...], optional): Shift the Tensor while padding. Defaults to :math:`0`.
        mode (str): Padding mode. Defaults to "constant".
        value (float): Padding value. Defaults to :math:`0`.
    """
    # offset gets subtracted from the left and added to the right
    offset = (torch.LongTensor([offset]) * torch.LongTensor([[-1], [1]])).flip(1)
    # compute the excess in each dim (negative val -> crop, positive val -> pad)
    excess = torch.Tensor([(size[-i] - x.shape[-i]) / 2 for i in range(1, len(size) + 1)])
    # floor excess on left side, ceil on right side, add offset
    pad = torch.stack([excess.floor(), excess.ceil()], dim=0).long() + offset

    return torch.nn.functional.pad(x, tuple(pad.T.flatten()), mode, value)
