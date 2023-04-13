import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Conv2dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, act_fn=nn.ReLU()):
        super(Conv2dRNNCell, self).__init__()

        padding = kernel_size // 2

        self.conv_xh = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv_hh = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)

        self.act_fn = act_fn

        # Initialize the hidden state parameter
        self.h = None

    def forward(self, x):
        # x: input sequence, shape (batch_size, input_size, sequence_length)

        # Compute the convolutional output
        c = self.conv_xh(x) + self.conv_hh(self.h)
        
        # Apply the non-linear activation function
        c = self.act_fn(c)
        
        # Update the hidden state
        self.h = c
        
        return c

    def reset_h(self, x_i, i):
        # h: hidden state, shape (batch_size, hidden_size, sequence_length)
        x_shape = x_i.shape
        x_shape = [x_shape[0], x_shape[1], x_shape[2]// (2**i), x_shape[3]// (2**i)]
        self.h = torch.randn(x_shape[0], self.conv_hh.out_channels, x_shape[2], x_shape[3])
        self.h = self.h.to(x_i.device)


# Taken from: https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
class Conv2dGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3) -> None:
        super(Conv2dGRUCell, self).__init__()

        self.h_init = False
        self.h_dim = hidden_size
        self.h = None

        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(in_channels=input_size + hidden_size, out_channels=2*hidden_size,
                                     kernel_size=kernel_size, padding=padding)
        self.conv_candidates = nn.Conv2d(in_channels=input_size + hidden_size, out_channels=hidden_size,
                                          kernel_size=kernel_size, padding=padding)

    def forward(self, x):

        # x shape: (batch, h_dim, h, w)
        
        combined = torch.cat([x, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.h_dim, dim=1)
        reset_gate = nn.Sigmoid()(gamma)
        update_gate = nn.Sigmoid()(beta)

        combined = torch.cat([x, reset_gate * self.h], dim=1)
        combined_cand = self.conv_candidates(combined)
        cnm = nn.Tanh()(combined_cand)

        self.h = (1 - update_gate) * self.h + update_gate * cnm

        return self.h
    
    def reset_h(self, x_i, i):
        # h: hidden state, shape (batch_size, hidden_size, sequence_length)
        x_shape = x_i.shape
        x_shape = [x_shape[0], x_shape[1], x_shape[2]// (2**i), x_shape[3]// (2**i)]
        self.h = torch.randn(x_shape[0], self.h_dim, x_shape[2], x_shape[3])
        self.h = self.h.to(x_i.device)
