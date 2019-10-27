import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.parameter import Parameter


class KernelGCN(nn.Module):
    def __init__(self, nfeat, nh1, nclass, dropout, bias=True):
        super(KernelGCN, self).__init__()
        self.w0 = Parameter(torch.FloatTensor(nfeat, nh1))
        self.w1 = Parameter(torch.FloatTensor(nh1, nclass))
        
        self.w = Parameter(torch.FloatTensor(3, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(nclass))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.fill_(1.0)
        nn.init.xavier_uniform_(self.w0, gain=1.414)
        nn.init.xavier_uniform_(self.w1, gain=1.414)
        
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, input, adjs):
        x = torch.mm(input, self.w0)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.mm(x, self.w1)
        
        
        adj = self.w[0]*adjs[0] + self.w[1]*adjs[1] + self.w[2] * adjs[2]
        
        feat = torch.mm(adj, x)
        if self.bias is not None:
            output = feat + self.bias

        return F.normalize(feat), F.log_softmax(output, dim=1)
