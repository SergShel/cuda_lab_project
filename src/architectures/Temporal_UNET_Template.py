import torch
import torch.nn as nn
from .temporal_modules import Conv2dGRUCell, Conv2dRNNCell
from .architecture_configs import *


class Temporal_UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        InitBlock: nn.Module,
        DownsampleBlock: nn.Module,
        block_dims: tuple[tuple[int, ...]],
        use_pooling: bool = False,
        batch_norm: bool = True,
        temporal_cell = Conv2dGRUCell,
    ) -> None:
        super(Temporal_UNetEncoder, self).__init__()

        self.in_block = InitBlock(block_dims[0], batch_norm)
        self.downsample_blocks = nn.ModuleList(
            [DownsampleBlock(channels, use_pooling, batch_norm) for channels in block_dims[1:]]
        )

        temporal_conv = []
        for channels in block_dims:
            in_size = channels[-1]
            temporal_conv.append(temporal_cell(input_size=in_size, hidden_size=in_size, kernel_size=3))

        self.temporal_conv = nn.ModuleList(temporal_conv)

    def freeze_temporal(self):
        self.temporal_conv = self.temporal_conv.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)

        skip_connect = []
        temporal_states = []
        for block, conv_temp_cell in zip(self.downsample_blocks, self.temporal_conv[:-1]):
            # skip_connect.append(x)
            # pass through RNN append rnn states for concatenation
            temporal_states.append(conv_temp_cell(x))
            x = block(x)

        # pass through the last conv rnn state
        x = self.temporal_conv[-1](x)

        return x, skip_connect, temporal_states


class Temporal_UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        UpsampleBlock: nn.Module,
        out_channels: int,
        block_dims: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        concat_hidden: bool = True,
    ) -> None:
        super(Temporal_UNetDecoder, self).__init__()

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

    def forward(self, x: torch.Tensor, temporal_states: list[torch.Tensor]) -> torch.Tensor:
        for block, temporal_conv in zip(self.upsample_blocks, reversed(temporal_states)):
            if self.concat_hidden:
                x = block(x, temporal_conv)
                # temporal_conv = center_pad(temporal_conv, x.shape[2:])
            else:
                # currently deprecated
                x = block.upsample(x)
                # temporal_conv = center_pad(temporal_conv, x.shape[2:])
                x = x + temporal_conv
                x = block.conv_block(x)

        return self.out_block(x)


class Temporal_UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: Temporal_TemplateUNetConfig) -> None:
        super(Temporal_UNet, self).__init__()

        self.config = config
        self.encoder = Temporal_UNetEncoder(
            config.initblock,
            config.downsampleblock,
            config.encoder_blocks[0],
            config.use_pooling,
            config.batch_norm,
            config.temporal_cell
        )
        self.decoder = Temporal_UNetDecoder(
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
        # loop over sequence
        outputs = []
        # reset hidden state for a new sequence
        for i, temp_conv_cell in enumerate(self.encoder.temporal_conv):
            temp_conv_cell.reset_h(x[:, 0], i)
        # x is Batch x Sequence x Channel x Height x Width
        for i in range(x.shape[1]):
            out, skip_connect, temporal_states = self.encoder(x[:, i])
            out = self.decoder(out, temporal_states)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs

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

    def freeze_temporal(self):
        self.encoder.freeze_temporal()
