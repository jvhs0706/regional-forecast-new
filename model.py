import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pk

from model_utils.lstm import Seq2seq, BidirectionalLSTM
from model_utils.utils import TimestepDense
from model_utils.broadcasting import *

from data_utils import *

import sys

class Regional(nn.Module):
    def __init__(self, source_station_obs_num_feat: dict, 
        dropout = 0.5, horizon_length = 24 * horizon_days,
        broadcasting = 'Broadcasting'):
        super().__init__()
        self.source_station_seq2seqs = nn.ModuleDict()
        for st, num_feat in source_station_obs_num_feat.items():
            self.source_station_seq2seqs[st] = Seq2seq(num_feat, 64, horizon_length, 64, dropout)
        self.broadcasting = eval(broadcasting)(num_source = len(source_station_obs_num_feat), num_step = horizon_length, num_feat = 64)
        self.process_wrf_cmaq = nn.Sequential(
            BidirectionalLSTM(len(wrf_species) + len(cmaq_species) + len(cmaq_summation_species), 32, 32, True),
            BidirectionalLSTM(64, 32, 32, True)
        )
        self.tail = nn.Sequential(
            BidirectionalLSTM(128, 32, 32, True),
            BidirectionalLSTM(64, 32, 32, True),
            TimestepDense(64, 2, horizon_length, dropout, True, None)
        )

    def forward(self, source_obs_dic: dict, target_wrf_cmaq: torch.Tensor, target_source_dist: torch.Tensor):
        '''
        source_obs_dic: (*B, C_s, T) for each source station
        target_wrf_cmaq: (*D_target, *B, C, T)
        target_source_dist: (*D_target, num_source)
        '''
        obs_out = self.broadcasting(torch.stack([self.source_station_seq2seqs[st](arr) for st, arr in source_obs_dic.items()]), target_source_dist)
        wrf_cmaq_out = self.process_wrf_cmaq(target_wrf_cmaq)
        out = self.tail(torch.cat([obs_out, wrf_cmaq_out], dim = -2))
        return out  

    def predict(self, source_obs_dic: dict, target_wrf_cmaq: torch.Tensor, target_source_dist: torch.Tensor, denormalizing):
        '''
        source_obs_dic: (*B, C_s, T) for each source station
        target_wrf_cmaq: (*D_target, *B, C, T)
        target_source_dist: (*D_target, num_source)
        '''
        means, stds = denormalizing
        assert not self.training
        with torch.no_grad():
            out = self(source_obs_dic, target_wrf_cmaq, target_source_dist).cpu().numpy()
            out = out * stds[:, None] + means[:, None]
        return out

if __name__ == '__main__':
    pass