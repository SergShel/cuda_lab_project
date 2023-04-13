from dataclasses import dataclass
from dataclasses import asdict
import torch
import torch.nn as nn
from . import vanilla_modules
from . import resnet_modules
from . import convnext_modules
from .temporal_modules import Conv2dRNNCell, Conv2dGRUCell
import json

@dataclass
class Config:
    """Configuration Class."""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            dict_wo_nones = {k: v for k, v in asdict(self).items() if v is not None}
            json.dump(dict_wo_nones, f, cls=CustomEncoder, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = cls(**json.load(f))
        return config

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return o.__name__
        return super().default(o)

@dataclass
class Temporal_TemplateUNetConfig(Config):
    """Configuration for U-Net."""
    # these are the dimensions for concatenation,
    # if summing is wanted, reduce the first dimension for each decoder block
    encoder_blocks: list[list[int]]
    decoder_blocks: list[list[int]]
    initblock: nn.Module
    downsampleblock: nn.Module
    upsampleblock: nn.Module
    temporal_cell: nn.Module
    out_channels: int = 20
    concat_hidden: bool = True
    use_pooling: bool = False
    batch_norm: bool = True

@dataclass
class Temporal_VanillaUNetConfig(Temporal_TemplateUNetConfig):
    initblock: nn.Module = vanilla_modules.ConvBlock
    downsampleblock: nn.Module = vanilla_modules.DownsampleBlock
    upsampleblock: nn.Module = vanilla_modules.UpsampleBlock
    temporal_cell: nn.Module = Conv2dGRUCell

@dataclass
class Temporal_ResUNetConfig(Temporal_TemplateUNetConfig):
    initblock: nn.Module = resnet_modules.ResNet18Initblock
    downsampleblock: nn.Module = resnet_modules.Resnet18Downblock
    upsampleblock: nn.Module = resnet_modules.Resnet18Upblock
    temporal_cell: nn.Module = Conv2dGRUCell

@dataclass
class Temporal_ConvUNextConfig(Temporal_TemplateUNetConfig):
    initblock: nn.Module = convnext_modules.ConvNextLikeInitblock
    downsampleblock: nn.Module = convnext_modules.ConvNextDownblock
    upsampleblock: nn.Module = convnext_modules.ConvNextUpblock
    temporal_cell: nn.Module = Conv2dGRUCell

@dataclass
class Original_Dimensions:
    # original dimensions of the proposed RNN Unet
    encoder_blocks: list[list[int]] = [[3, 64, 64], [64, 128, 128], [128, 256, 256]],
    decoder_blocks: list[list[int]] = [[512, 256, 256], [256, 128, 128], [128, 64, 64]],

@dataclass
class SmallShallow_NetworkSize:
    # maps 64 to 16, 128 to 32, 256 to 64, 512 to 128
    encoder_blocks: list[list[int]] = [[3, 16, 16], [16, 32, 32], [32, 64, 64]],
    decoder_blocks: list[list[int]] = [[128, 64, 64], [64, 32, 32], [32, 16, 16]],

@dataclass
class SmallDeep_NetworkSize:
    # maps 64 to 16, 128 to 32, 256 to 64, 512 to 128
    encoder_blocks: list[list[int]] = [[3, 16, 16], [16, 32, 32], [32, 64, 64], [64, 128, 128]],
    decoder_blocks: list[list[int]] = [[256, 128, 128], [128, 64, 64], [64, 32, 32], [32, 16, 16]],
