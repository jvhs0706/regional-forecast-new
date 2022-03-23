import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pk

from model_utils.lstm import Seq2seq, BidirectionalLSTM
from model_utils.utils import TimestepDense

from data_utils.utils import *

class LSTM_FC(nn.Module):
    def __init__(self, source_station_obs_num_feat: int, 
        dropout = 0.5, horizon_length = 24 * horizon_days):
        super().__init__()
        
        self.seq2seq = Seq2seq(source_station_obs_num_feat, 64, horizon_length, 64, dropout)
        self.process_wrf_cmaq = BidirectionalLSTM(len(wrf_species) + len(cmaq_species) + len(cmaq_summation_species), 32, 32, True)

        self.tail = nn.Sequential(
            BidirectionalLSTM(128, 32, 32, True),
            TimestepDense(64, 2, horizon_length, dropout, True, None)
        )

    def forward(self, obs: torch.Tensor, wrf_cmaq: torch.Tensor):
        '''
        source_obs_dic: (*B, C_s, T)
        target_wrf_cmaq: (*B, C, T)
        '''
        obs_out = self.seq2seq(obs)
        wrf_cmaq_out = self.process_wrf_cmaq(wrf_cmaq)
        out = self.tail(torch.cat([obs_out, wrf_cmaq_out], dim = -2))
        return out  

    def predict(self, obs: dict, wrf_cmaq: torch.Tensor, denormalizing):
        '''
        source_obs_dic: (*B, C_s, T) for each source station
        target_wrf_cmaq: (*B, C, T)
        '''
        means, stds = denormalizing
        assert not self.training
        with torch.no_grad():
            out = self(obs, wrf_cmaq).cpu().numpy()
            out = out * stds[:, None] + means[:, None]
        return out