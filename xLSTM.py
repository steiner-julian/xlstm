import math
import torch

from torch import nn
from torch.nn import LSTM

class sLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        drouput,
        bidirectional,
        proj_size
    ):
        pass

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super(mLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.Wq = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bq = nn.Parameter(torch.randn(hidden_size, 1))
        self.Wk = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bk = nn.Parameter(torch.randn(mem_dim, 1))
        self.Wv = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bv = nn.Parameter(torch.randn(mem_dim, 1))
        self.wi = nn.Parameter(torch.randn(1, input_size))
        self.bi = nn.Parameter(torch.randn(1))
        self.wf = nn.Parameter(torch.randn(1, input_size))
        self.bf = nn.Parameter(torch.randn(1))
        self.Wo = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bo = nn.Parameter(torch.randn(hidden_size, 1))
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, states):
        (C_prev, n_prev) = states
        qt = torch.matmul(self.Wq, x) + self.bq
        kt = (1 / math.sqrt(self.mem_dim)) * (torch.matmul(self.Wk, x) + self.bk)
        vt = torch.matmul(self.Wv, x) + self.bv

        it = torch.exp(torch.matmul(self.wi, x) + self.bi)
        ft = torch.sigmoid(torch.matmul(self.wf, x) + self.bf)

        vt = vt.squeeze()
        kt = kt.squeeze()

        C = ft * C_prev + it * torch.ger(vt, kt)
        n = ft * n_prev + it * kt.unsqueeze(1)

        max_nqt = torch.max(torch.abs(torch.matmul(n.T, qt)), torch.tensor(1.0))
        h_tilde = torch.matmul(C, qt) / max_nqt
        ot = torch.sigmoid(torch.matmul(self.Wo, x) + self.bo)
        ht = ot * h_tilde

        return ht, (C, n)

    def init_hidden(self):
        return (torch.zeros(self.mem_dim, self.mem_dim),
                torch.zeros(self.mem_dim, 1))


class xLSTM(nn.Module):
    def __init__(self):
        pass