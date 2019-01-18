import torch
import torch.nn as nn
import torch.nn.init as init

def init_linear(m):
    return init(m,nn.init.orthogonal_,lambda x : nn.init.constant_(x,0),nn.init.calculate_gain('relu'))

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_gru(module,):
    nn.init.orthogonal_(module.weight_ih.data)
    nn.init.orthogonal_(module.weight_hh.data)
    module.bias_ih.data.fill_(0)
    module.bias_hh.data.fill_(0)

def init_lstm(lstm,bidirectional=False,layers=1):
    init.orthogonal_(lstm.weight_ih_l0.data)
    init.orthogonal_(lstm.weight_hh_l0.data)
    init.constant_(lstm.bias_ih_l0.data,val=0)
    init.constant_(lstm.bias_hh_l0.data,val=0)
    lstm.bias_ih_l0.data.chunk(4)[1].fill_(1)

    if bidirectional:
        init.orthogonal_(lstm.weight_ih_l0_reverse.data)
        init.orthogonal_(lstm.weight_hh_l0_reverse.data)
        init.constant_(lstm.bias_ih_l0_reverse.data, val=0)
        init.constant_(lstm.bias_hh_l0_reverse.data, val=0)
        lstm.bias_ih_l0_reverse.data.chunk(4)[1].fill_(1)

    if layers == 2:
        init.orthogonal_(lstm.weight_ih_l1.data)
        init.orthogonal_(lstm.weight_hh_l1.data)
        init.constant_(lstm.bias_ih_l1.data, val=0)
        init.constant_(lstm.bias_hh_l1.data, val=0)
        lstm.bias_ih_l1.data.chunk(4)[1].fill_(1)

        if bidirectional:
            init.orthogonal_(lstm.weight_ih_l1_reverse.data)
            init.orthogonal_(lstm.weight_hh_l1_reverse.data)
            init.constant_(lstm.bias_ih_l1_reverse.data, val=0)
            init.constant_(lstm.bias_hh_l1_reverse.data, val=0)
            lstm.bias_ih_l1_reverse.data.chunk(4)[1].fill_(1)


