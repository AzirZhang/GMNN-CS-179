from torch import nn
import torch.nn.functional as F
from layer import GraphConvolution


class GNNq(nn.Module):

    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['hidden_dim'])])
        self.m3 = GraphConvolution(opt_, adj)

        # opt_ = dict([('in', 2*opt['hidden_dim']), ('out', opt['hidden_dim'])])
        # self.m4 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()
        self.m3.reset_parameters()
        # self.m4.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m3(x)
        x = F.relu(x)
        # x = F.dropout(x, self.opt['dropout'], training=self.training)
        # x = self.m4(x)
        # x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x


class GNNp(nn.Module):

    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['hidden_dim'])])
        self.m3 = GraphConvolution(opt_, adj)

        # opt_ = dict([('in', 2*opt['hidden_dim']), ('out', opt['hidden_dim'])])
        # self.m4 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()
        self.m3.reset_parameters()
        # self.m4.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m3(x)
        x = F.relu(x)
        # x = F.dropout(x, self.opt['dropout'], training=self.training)
        # x = self.m4(x)
        # x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x
        