import torch

from .SequenceTagger import SequenceTagger
from .evaluation import FScore

class BaseSegmenter(SequenceTagger):
    def __init__(self,label_vocab,**kwargs):
        super(BaseSegmenter, self).__init__(**kwargs)

        self.label_vocab = label_vocab
    def greedyDecode(self,logits,lengths):
        label_indexs = torch.argmax(logits[:,:,2:],dim=-1)+2
        decoded = []
        for i in range(len(lengths)):
            label_idxs = label_indexs[i][:lengths[i]]
            decoded.append(label_idxs)
        return decoded

    def train_epoch(self,train_iter):

        self.train()
        total_batch = len(train_iter)
        cur_batch = 1
        loss = 0.0

        for batch in train_iter:
            unigrams,unigrams_length = batch.unigram
            fwd_bigrams,_ = batch.fwd_bigram
            back_bigrams,_ = batch.back_bigram
            labels= batch.label

            logits = self.forward(unigrams,unigrams_length,fwd_bigrams,back_bigrams)
            batch_loss = self.loss_func(logits.permute(0,2,1),labels)
            loss += batch_loss.item()
            self.update(batch_loss)
            cur_batch += 1
            print ('\rBatch {}/{}, Training Loss:{:.2f}'.format(cur_batch,total_batch,loss/cur_batch),end='')

    def evaluate(self,iter):
        self.eval()
        fscore = FScore()
        decoded = self.decode(iter)
        gold = [example.label for example in iter.dataset.examples]
        assert len(decoded) == len(gold)
        for d,g in zip(decoded,gold):
            fscore += FScore.evaluate_BMSE(g,d)
        return  fscore

    def to_label(self,label_idxs):
        return list(map(lambda x:self.label_vocab.itos[x],label_idxs))

    def decode(self,data):
        self.eval()

        decoded = []

        for batch in data:
            unigrams,unigrams_length = batch.unigram
            fwd_bigrams,_ =batch.fwd_bigram
            back_bigrams,_ = batch.back_bigram
            logits = self.forward(unigrams,unigrams_length,fwd_bigrams,back_bigrams)
            decoded_idx = self.greedyDecode(logits,unigrams_length)
            decoded.extend(list(map(lambda x:self.to_label(x),decoded_idx)))

        return decoded

