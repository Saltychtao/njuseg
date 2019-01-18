import torch
import torch.nn as nn
import torch.nn.init as init


from .init import init_lstm
from .module import Classifier,LSTMEncoder



class SequenceTagger(nn.Module):
    def __init__(self,
                 token_embedding_dim,
                 hidden_dim,
                 lstm_hidden_dim,
                 clf_hidden_dim,
                 token_vocab_size,
                 label_size,
                 pad_idx=None,
                 lstm_layer=2,
                 token_pretrained=None,
                 subtoken=False,
                 subtoken_embedding_dim=50,
                 subtoken_pretrained=None,
                 subtoken_vocab_size=None,
                 freeze_pretrained=False,
                 dropout=0.5):

        super(SequenceTagger, self).__init__()

        self.pad_idx = pad_idx
        self.label_size = label_size
        self.lstm_layer = lstm_layer
        self.subtoken = subtoken

        if token_pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(token_pretrained,freeze_pretrained)
        else:
            self.embeddings = nn.Embedding(token_vocab_size,token_embedding_dim,padding_idx=pad_idx)

        if subtoken:
            if subtoken_pretrained is not None:
                self.subtoken_embeddings = nn.Embedding.from_pretrained(subtoken_pretrained,freeze_pretrained)
            else:
                self.subtoken_embeddings = nn.Embedding(subtoken_vocab_size,subtoken_embedding_dim,padding_idx=pad_idx)

        if subtoken:
            self.emb2hidden = nn.Sequential(
                nn.Linear(token_embedding_dim+2*subtoken_embedding_dim,hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
        else:
            self.emb2hidden = nn.Sequential(
                nn.Linear(token_embedding_dim,hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )

        # self.lstm = nn.LSTM(input_size=hidden_dim,
        #                     hidden_size=hidden_dim,
        #                     num_layers=lstm_layer,
        #                     bidirectional=True,
        #                     batch_first=True,
        #                     dropout=dropout
        #                     )

        self.encoder = LSTMEncoder(input_size=hidden_dim,
                                   hidden_size=lstm_hidden_dim,
                                   num_layers=lstm_layer,
                                   dropout=dropout,
                                   batch_first=True,
                                   bidirectional=True
                                   )

        self.classifier = Classifier(label_size,2*lstm_hidden_dim ,clf_hidden_dim,1,use_batchnorm=False,dropout_prob=dropout)

        self.loss_func = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(p=dropout)

        self.optimizer = torch.optim.Adam(params=self.parameters(),
                                          lr=5e-4,)

    def reset_parameters(self):
        if self.pad_idx is not None:
            self.embeddings.weight[self.pad_idx].fill_(0)
            if self.subtoken:
                self.subtoken_embeddings.weight[self.pad_idx].fill_(0)
        init_lstm(self.lstm,bidirectional=True,layers=self.lstm_layer)
        self.classifier.reset_parameters()

    def forward(self, tokens,lengths,subtokens=None,back_subtokens =None):
        embs = self.embeddings(tokens)
        if self.subtoken:
            subtoken_embs = self.subtoken_embeddings(subtokens)
            back_subtokens = self.subtoken_embeddings(back_subtokens)
            embs = torch.cat([embs,subtoken_embs,back_subtokens],dim=-1)
        embs = self.emb2hidden(embs)

        # lengths, perm_idx = lengths.sort(0, descending=True)
        # embs = embs[perm_idx]
        #
        # embs = pack(embs, lengths, batch_first=True)
        # output, hidden = self.lstm(embs)
        # output = unpack(output,batch_first=True)[0]
        #
        # _, unperm_idx = perm_idx.sort(0)
        # output = output[unperm_idx]

        outputs,_ = self.encoder(embs,lengths,need_sort=True)

        logits = self.classifier(outputs.contiguous())

        return logits

    def update(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
