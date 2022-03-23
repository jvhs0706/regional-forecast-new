import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .dense import *

class LSTMEncoder(nn.Module):
    def __init__(self, in_channels: int, encoded_size: int):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(in_channels, encoded_size)

    def forward(self, x):
        N, C, T = x.shape 
        assert C == self.lstm_cell.input_size
        C, H = self.lstm_cell.input_size, self.lstm_cell.hidden_size
        out, c = torch.zeros((N, H), dtype = x.dtype, device = x.device), torch.zeros((N, H), dtype = x.dtype, device = x.device)
        for t in range(T):
            out, c = self.lstm_cell(x[:, :, t], (out, c))
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, encoded_size: int, length: int):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(0, encoded_size)
        self.length = length 

    def forward(self, h_in):
        N, H = h_in.shape
        assert H == self.lstm_cell.hidden_size
        out, c_in = [], torch.zeros((N, H), dtype = h_in.dtype, device = h_in.device)
        
        for t in range(self.length):
            if t == 0:
                h, c = self.lstm_cell(torch.zeros((N, 0), dtype = h_in.dtype, device = h_in.device), (h_in, c_in))
            else:
                h, c = self.lstm_cell(torch.zeros((N, 0), dtype = h_in.dtype, device = h_in.device), (h, c))
            out.append(h)

        out = torch.stack(out, axis = -1)
        return out

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, forward_output_size: int, backward_output_size: int, batch_norm: bool):
        super().__init__()
        self.flstm_cell = nn.LSTMCell(input_size, forward_output_size)
        self.blstm_cell = nn.LSTMCell(input_size, backward_output_size)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(forward_output_size + backward_output_size)

    def forward(self, x):
        C, T = x.shape[-2:] 
        fH, bH = self.flstm_cell.hidden_size, self.blstm_cell.hidden_size
        fout, bout = [], []
    
        for t in range(T):
            if t == 0:
                h, c = self.flstm_cell(x[..., t].view(-1, C))
            else:
                h, c = self.flstm_cell(x[..., t].view(-1, C), (h, c))
            fout.append(h)
        fout = torch.stack(fout, axis = -1)
    
        for t in reversed(range(T)):
            if t == T-1:
                h, c = self.blstm_cell(x[..., t].view(-1, C))
            else:
                h, c = self.blstm_cell(x[..., t].view(-1, C), (h, c))
            bout.append(h)
        bout = torch.stack(bout[::-1], axis = -1)
        
        out = torch.cat([fout, bout], axis = -2)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        return out.view(*x.shape[:-2], fH + bH, T)


class Seq2seq(nn.Module):
    def __init__(self, in_channels: int, encoded_size: int, num_step: int, out_channels: int, dropout: float):
        super().__init__()
        self.encoder = LSTMEncoder(in_channels = in_channels, encoded_size = encoded_size)
        self.decoder = LSTMDecoder(encoded_size = encoded_size, length = num_step)
        self.dense = TimestepDense(in_features = encoded_size, out_features = out_channels, num_step = num_step, dropout = dropout, time_distributed = False)
    
    def forward(self, x):
        out = self.decoder(self.encoder(x.view(-1, *x.shape[-2:])))
        out = self.dense(out.view(*x.shape[:-2], *out.shape[-2:]))
        return out

if __name__ == '__main__':
    pass