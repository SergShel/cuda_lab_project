import torch
import torch.nn as nn


def get_act_fn(act_name: str, slope=0.2):
    if act_name == "relu":
        return nn.ReLU()
    elif act_name == "leakyrelu":
        return nn.LeakyReLU(slope)
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()


class ConvBlock(nn.Module):
    """
    Encapuslation of a convolutional block (conv + activation + pooling)
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad=None, pool=False, batchNorm=False, act_name="relu", dropout=None):
        super().__init__()
        act_fn = get_act_fn(act_name)
        if pad==None:
            pad = kernel_size//2
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad))
        if batchNorm == True:
            layers.append(nn.BatchNorm2d(out_channel))
        layers.append(act_fn)
        if pool == True:
            layers.append(nn.MaxPool2d(kernel_size=2))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class ConvUpsampleBlock(nn.Module):
    """
    Encapuslation of a convolutional block (conv + activation + pooling)
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, upsample_mode="nearest", batchNorm=False, act_name="relu", dropout=None):
        super().__init__()
        act_fn = get_act_fn(act_name)
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
        layers.append(nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size//2
            ))
        if batchNorm == True:
            layers.append(nn.BatchNorm2d(out_channel))
        layers.append(act_fn)
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class ConvTransposeBlock(nn.Module):
    """
    Simple convolutional block: ConvTranspose + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y

class MLP(nn.Module):
    
    def __init__(self, hidden_size, act_fn_name="relu", activation=True, batchnorm=False):
        super(MLP, self).__init__()
        act_fn = get_act_fn(act_fn_name)
        layers = []
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            if batchnorm==True:
                layers.append(nn.BatchNorm1d(hidden_size[i+1]))
            if activation==True:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, upsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.upsample = upsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = x.clone()
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)
        return out

