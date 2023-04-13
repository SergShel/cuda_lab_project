import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union
from pathlib import Path
from architectures.vanilla_modules import * 
from architectures.temporal_modules import *


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        InitBlock: nn.Module,
        DownsampleBlock: nn.Module,
        block_dims: tuple[tuple[int, ...]],
        use_pooling: bool = False,
        batch_norm: bool = True,
    ) -> None:
        super(UNetEncoder, self).__init__()

        self.in_block = InitBlock(block_dims[0], batch_norm)
        self.downsample_blocks = nn.ModuleList(
            [DownsampleBlock(channels, use_pooling, batch_norm) for channels in block_dims[1:]]
        )

       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)    

        hidden_states = []
        for block in self.downsample_blocks:
            hidden_states.append(x)
            x = block(x)

        return x, hidden_states

class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        UpsampleBlock: nn.Module,
        out_channels: int,
        block_dims: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        concat_hidden: bool = True,
    ) -> None:
        super(UNetDecoder, self).__init__()

        self.upsample_blocks = nn.ModuleList(
            [UpsampleBlock(channels, batch_norm, concat_hidden) for channels in block_dims[1:]]
        )
        
        self.out_block = nn.Conv2d(
            in_channels=block_dims[-1][-1],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.concat_hidden = concat_hidden

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        for block, h in zip(self.upsample_blocks, reversed(hidden_states)):
            if self.concat_hidden:
                x = block.upsample(x)
                h = center_pad(h, x.shape[2:])
                x = torch.cat([x, h], dim=1)
                x = block.conv_block(x)
            else:
                x = block.upsample(x)
                h = center_pad(h, x.shape[2:])
                x = x + h
                x = block.conv_block(x)
        return self.out_block(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config) -> None:
        super(UNet, self).__init__()

        self.config = config
        self.encoder = UNetEncoder(
            config.initblock,
            config.downsampleblock,
            config.encoder_blocks[0], 
            config.use_pooling, 
            config.batch_norm,
        )
        self.decoder = UNetDecoder(
            config.upsampleblock,
            config.out_channels,
            config.decoder_blocks[0],
            config.batch_norm,
            config.concat_hidden,
        )

        # self.encoder.apply(init_weights)
        # self.decoder.apply(init_weights)
        self.out_block_in_channels = config.decoder_blocks[0][-1][-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        x = x.unsqueeze(1)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def replace_outchannels(self,  out_channels):
        # replaces the last layer of the outchannels, such that the model can be adjusted to only 
        # output a certain amount of layers
        self.decoder.out_block = nn.Conv2d(
            in_channels=self.out_block_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    # @classmethod
    # def from_pretrained(cls, path: str, config: Union[UNetConfig, str, None] = None) -> "UNet":
    #     path = Path(path)

    #     if config is None:
    #         config = UNetConfig.from_file(path.parent / "model.json")
    #     elif not isinstance(config, UNetConfig):
    #         config = UNetConfig.from_file(config)

    #     model = cls(config)
    #     model.load_state_dict(torch.load(path))
    #     return model

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