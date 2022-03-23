import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .dense import *
from .lstm import *

class Broadcasting(nn.Module):
    def __init__(self, num_source: int, **kwargs):
        super().__init__()
        self.distance_decay = nn.Parameter(torch.zeros(num_source)) # (num_source, )
    
    def forward(self, X: torch.Tensor, dist: torch.Tensor):
        '''
        X: (num_source, *)
        dist: (*D_target, num_source)
        '''
        weight = F.softmax(-self.distance_decay * dist, dim = -1)
        out = weight @ X.view(X.shape[0], -1) # 
        return out.view(*dist.shape[:-1], *X.shape[1:])

class BroadcastingTimeDistributed(nn.Module):
    def __init__(self, num_source: int, num_step: int, **kwargs):
        super().__init__()
        self.distance_decay = nn.Parameter(torch.zeros(num_source, num_step)) # (num_source, num_step)
    
    def forward(self, X: torch.Tensor, dist: torch.Tensor):
        '''
        X: (num_source, *, T)
        dist: (*D_target, num_source)
        '''
        weight = F.softmax(-self.distance_decay * dist.unsqueeze(-1), dim = -2)
        out = weight.view(-1, *weight.shape[-2:]).permute(2, 0, 1) @ X.view(X.shape[0], -1, X.shape[-1]).permute(2, 0, 1)
        return out.permute(1, 2, 0).view(*dist.shape[:-1], *X.shape[1:])

class BroadcastingSelfAttention(nn.Module):
    def __init__(self, num_source: int, num_feat: int, **kwargs):
        super().__init__()
        self.attention_weight = nn.Parameter(torch.empty(num_source, num_feat)) # (num_source, num_feat)
        self.attention_bias = nn.Parameter(torch.empty(num_source))
        nn.init.uniform_(self.attention_weight, -np.sqrt(6/num_feat), +np.sqrt(6/num_feat))
        nn.init.constant_(self.attention_bias, 0.0)

    def forward(self, X: torch.Tensor, dist: torch.Tensor):
        '''
        X: (num_source, *, num_feat, T)
        dist: (*D_target, num_source)
        '''
        dist_decay_weight = F.softmax(-F.softplus(torch.sum(X.view(X.shape[0], -1, *X.shape[-2:]) * self.attention_weight[:, None, :, None], axis = -2) + self.attention_bias[:, None, None]) * dist[..., None, None], dim = -3)
        out = dist_decay_weight.view(-1, *dist_decay_weight.shape[-3:]).permute(2, 3, 0, 1) @ X.view(X.shape[0], -1, *X.shape[-2:]).permute(1, 3, 0, 2)
        return out.permute(2, 0, 3, 1).view(*dist.shape[:-1], *X.shape[1:])

if __name__ == '__main__':
    pass
