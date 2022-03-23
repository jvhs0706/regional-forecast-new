import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class TimeDistributedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_step: int):
        super().__init__()
        self.in_features, self.out_features, self.num_step = in_features, out_features, num_step
        weight, bias = torch.empty(out_features, in_features, num_step), torch.empty(out_features, num_step)
        self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)
        nn.init.uniform_(self.weight, -np.sqrt(6/(in_features+out_features)), +np.sqrt(6/(in_features+out_features)))
        nn.init.constant_(self.bias, 0.0)
    def forward(self, x):
        # input: (*, in_features, num_step)
        out = torch.matmul(x.view(-1, *x.shape[-2:]).permute(2, 0, 1), self.weight.permute(2, 1, 0)).permute(1, 2, 0)
        out += self.bias[None, :, :]
        return out.view(*x.shape[:-2], *out.shape[-2:])

class TimestepDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_step: int, dropout: float, time_distributed: bool, activation = F.relu):
        super().__init__()
        self.time_distributed = time_distributed
        if 0.0 < dropout < 1.0:
            self.dropout = nn.Dropout(p = dropout)

        if self.time_distributed:
            self.tdl = TimeDistributedLinear(in_features, out_features, num_step)
        else:
            self.linear = nn.Linear(in_features, out_features)

        if activation is not None:
            self.activation = activation
        
    def forward(self, x):
        '''
        x: (*, in_features, num_step)
        '''
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        if self.time_distributed:
            out = self.tdl(x)
        else:
            out = self.linear(x.view(-1, *x.shape[-2:]).transpose(-1, -2)).transpose(-1, -2)
            out = out.view(*x.shape[:-2], *out.shape[-2:])
        
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out

if __name__ == '__main__':
    pass