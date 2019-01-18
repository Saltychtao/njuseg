import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Classifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers,
                 use_batchnorm, dropout_prob):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        if use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        mlp_layers = []
        for i in range(num_layers):
            layer_in_features = hidden_dim if i > 0 else input_dim
            linear_layer = nn.Linear(in_features=layer_in_features,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            mlp_layer = nn.Sequential(linear_layer, relu_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        for i in range(self.num_layers):
            linear_layer = self.mlp[i][0]
            init.kaiming_normal_(linear_layer.weight.data)
            init.constant_(linear_layer.bias.data, val=0)
        init.uniform_(self.clf_linear.weight.data, -0.002, 0.002)
        init.constant_(self.clf_linear.bias.data, val=0)

    def forward(self, sentence):
        mlp_input = sentence
        if self.use_batchnorm:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.use_batchnorm:
            mlp_output = self.bn_mlp_output(mlp_output)
        # mlp_output = self.dropout(mlp_output)
        logits = self.clf_linear(mlp_output)
        return logits

class LSTMEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 batch_first,
                 bidirectional):
        super(LSTMEncoder, self).__init__()
        input_size = input_size
        dropout = 0 if num_layers == 1 else dropout
        self.rnn = nn.LSTM(input_size = input_size,hidden_size=hidden_size,
                           num_layers=num_layers,dropout=dropout,batch_first=batch_first,
                           bidirectional=bidirectional)

    def forward(self,inputs,lengths,need_sort=False):
        if need_sort:
            lengths,perm_idx = lengths.sort(0,descending=True)
            inputs = inputs[perm_idx]
        bsize = inputs.size()[0]
        # state_shape = self.config.n_cells,bsize,self.config.d_hidden
        # h0 = c0 = inputs.new_zeros(state_shape)
        inputs = pack(inputs,lengths,batch_first=True)
        outputs,(ht,ct) = self.rnn(inputs)
        outputs,_ = unpack(outputs,batch_first=True)

        if need_sort:
            _,unperm_idx = perm_idx.sort(0)
            outputs = outputs[unperm_idx]
        return outputs,ht.permute(1,0,2).contiguous().view(bsize,-1)


class BOWEncoder(nn.Module):

    def __init__(self,config):
        super(BOWEncoder,self).__init__()

        self.config = config

    def forward(self,inputs,lengths):
        return torch.div(inputs.sum(1),lengths.float().unsqueeze(-1))




