import argparse
import os
import dill

import torch
from torchtext import  datasets,data
from torchtext.data.example import Example


from model.BaseSegmenter import BaseSegmenter

from model.evaluation import FScore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Segmenter(object):

    def __init__(self):
        pass

    @staticmethod
    def load_dataset(unigram_field,bigram_field,label_field,batch_size,data_dir):
        train = datasets.SequenceTaggingDataset(path=os.path.join(data_dir,'train.tsv'),
                                                 fields=[('unigram',unigram_field),('label',label_field),('fwd_bigram',bigram_field),('back_bigram',bigram_field)],
                                                 )
        dev = datasets.SequenceTaggingDataset(path=os.path.join(data_dir,'dev.tsv'),
                                              fields=[('unigram',unigram_field),('label',label_field),('fwd_bigram',bigram_field),('back_bigram',bigram_field)])
        unigram_field.build_vocab(train,dev,min_freq=1)
        bigram_field.build_vocab(train,dev,min_freq=5)
        label_field.build_vocab(train,dev)

        train_iter = data.BucketIterator(train,
                                         train=train,
                                         batch_size=batch_size,
                                         sort_key=lambda x:len(x.unigram),
                                         device=device,
                                         sort_within_batch=True,
                                         repeat=False,
                                         )

        dev_iter = data.BucketIterator(dev,
                                       batch_size=32,
                                       device=device,
                                       sort=False,
                                       shuffle=False,
                                       repeat=False)

        return train_iter,dev_iter

    @staticmethod
    def load_testset(unigram_field,bigram_field,label_field,data_dir):
        test = datasets.SequenceTaggingDataset(path=os.path.join(data_dir,'test.tsv'),
                                               fields=[('unigram',unigram_field),('label',label_field),('fwd_bigram',bigram_field),('back_bigram',bigram_field)])

        test_iter = data.BucketIterator(test,batch_size=32,train=False,shuffle=False,sort=False,device=device)
        return test_iter

    @staticmethod
    def load_pretrained(filepath,vocab,dim):

        import numpy as np
        vecs = np.random.normal(0,1,(len(vocab),dim))
        with open(filepath,'r') as f:
            f.readline()
            for line in f:
                splited = line.split()
                word = splited[0]
                if word not in vocab.stoi:
                    continue
                vec = [float(f) for f in splited[1:]]
                vecs[vocab.stoi[word]] = vec
        tensor= torch.from_numpy(vecs)

        return tensor.float()


    @staticmethod
    def train(options):
        unigram_field = data.Field(include_lengths=True,batch_first=True,stop_words=None)
        bigram_field = data.Field(include_lengths=True,batch_first=True,stop_words=None)
        label_field = data.Field(batch_first=True,)
        train_iter, dev_iter = Segmenter.load_dataset(unigram_field,bigram_field, label_field, options.batch_size,options.data_dir)

        if 'token_pretrained' in options:
            token_pretrained = Segmenter.load_pretrained(options.token_pretrained,unigram_field.vocab,100)
        else:
            token_pretrained = None

        if 'subtoken_pretrained' in options:
            subtoken_pretrained = Segmenter.load_pretrained(options.subtoken_pretrained,bigram_field.vocab,100)
        else:
            subtoken_pretrained = None

        model = BaseSegmenter(
            label_field.vocab,
            token_embedding_dim=options.unigram_dim,
            lstm_hidden_dim=options.lstm_hidden_dim,
            hidden_dim=options.hidden_dim,
            clf_hidden_dim=options.clf_hidden_dim,
            token_vocab_size=len(unigram_field.vocab),
            label_size=len(label_field.vocab),
            pad_idx=unigram_field.vocab.stoi[unigram_field.pad_token],
            lstm_layer=options.lstm_layer,
            subtoken_embedding_dim=options.bigram_dim,
            token_pretrained=token_pretrained,
            subtoken_pretrained=subtoken_pretrained,
            subtoken=True,
            subtoken_vocab_size=len(bigram_field.vocab),
            freeze_pretrained=options.freeze_pretrained,
            dropout=options.dropout
            ).to(device)

        best_fscore = FScore()
        patience = options.patience
        for epoch in range(options.epoch):
            model.train_epoch(train_iter)
            dev_fscore = model.evaluate(dev_iter)
            print(' \nEpoch {}, Dev FScore : {}'.format(epoch,dev_fscore))
            patience -= 1
            if patience == 0:
                print('Out of patience...')
                exit()

            if dev_fscore > best_fscore:
                best_fscore = dev_fscore
                patience = options.patience
                Segmenter.save_model(model,options,fields={'unigram_field':unigram_field,'bigram_field':bigram_field,'label_field':label_field})
                print(' Model Saved to {}'.format(options.model_pth))


    @staticmethod
    def save_model(model,options,fields):
        dic = {}
        dic['state_dict'] = model.state_dict()
        dic['options'] = options
        for name,f in fields.items():
            dic[name] = f
        torch.save(dic,options.model_pth,pickle_module=dill)


    @staticmethod
    def load_model(model_pth,use_gpu=False):
        checkpoint = torch.load(model_pth, pickle_module=dill,map_location='cpu' if not use_gpu else None)
        unigram_field = checkpoint['unigram_field']
        bigram_field = checkpoint['bigram_field']
        label_field = checkpoint['label_field']
        options = checkpoint['options']

        device = torch.device('cuda' if use_gpu else 'cpu')

        model = BaseSegmenter(
            label_field.vocab,
            token_embedding_dim=options.unigram_dim,
            lstm_hidden_dim=options.lstm_hidden_dim,
            hidden_dim=options.hidden_dim,
            clf_hidden_dim=options.clf_hidden_dim,
            token_vocab_size=len(unigram_field.vocab),
            label_size=len(label_field.vocab),
            pad_idx=unigram_field.vocab.stoi[unigram_field.pad_token],
            lstm_layer=options.lstm_layer,
            subtoken_embedding_dim=options.bigram_dim,
            subtoken=True,
            subtoken_vocab_size=len(bigram_field.vocab),
            dropout=options.dropout
        ).to(device)

        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded Model from {}'.format(options.model_pth))
        segmenter = Segmenter()
        segmenter.model = model
        segmenter.unigram_field =unigram_field
        segmenter.bigram_field = bigram_field
        segmenter.label_field = label_field
        segmenter.options = options

        return segmenter

    @staticmethod
    def test(model_pth):

        segmenter = Segmenter.load_model(model_pth,device)
        test_iter = Segmenter.load_testset(segmenter.unigram_field,segmenter.bigram_field,segmenter.label_field,segmenter.options.data_dir)
        fscore = segmenter.model.evaluate(test_iter)
        return fscore

    @staticmethod
    def BMSE2seg(sentences,labels):
        ret = []
        for sent,label in zip(sentences,labels):
            example = ''
            for c,l in zip(sent,label):
                if l == 'S':
                    example += c + ' '
                elif l == 'B':
                    example += c
                elif l == 'M':
                    example += c
                elif l == 'E':
                    example += c + ' '
            ret.append(example.rstrip())
        return ret

    @staticmethod
    def seg_corpus(model_pth,raw_corpus,output,use_gpu=True):
        # device = torch.device('cuda' if use_gpu else 'cpu')
        import time
        start_time = time.time()
        segmenter  = Segmenter.load_model(model_pth,use_gpu)
        line_cnt = 0
        CHUNK_SIZE = 10000
        report_interval = 10000
        fout = open(output,'w')
        with open(options.raw_corpus,'r') as f:
            while True:
                cnt = 0
                raw_lines = []
                for line in f:
                    if line.strip() == '':
                        continue
                    raw_lines.append(line.strip())
                    cnt += 1
                    line_cnt += 1
                    if cnt >= CHUNK_SIZE:
                        break
                segmented = segmenter.seg(raw_lines)
                fout.write('\n'.join(segmented))
                print('\rProcessed {} lines'.format(line_cnt),end='')
                if cnt < CHUNK_SIZE:
                    break
        end_time = time.time()
        print('Total Elapsed Time: {:.2f}\n Total Line Processed: {}'.format(end_time-start_time,line_cnt))

    def seg(self,sentences):
        examples = []
        fields=[('unigram', self.unigram_field), ('fwd_bigram', self.bigram_field),('back_bigram', self.bigram_field)]
        for sent in sentences:
            columns = [[], [], []]
            chars = ['<BOS>'] + list(sent) + ['<EOS>']
            for c,f_bi,b_bi in zip(chars[1:-1],zip(chars,chars[1:]),zip(chars[1:],chars[2:])):
                fwd_bi = ''.join(f_bi)
                back_bi = ''.join(b_bi)
                columns[0].append(c)
                columns[1].append(fwd_bi)
                columns[2].append(back_bi)
            examples.append(Example.fromlist(columns,fields))

        dataset = data.Dataset(examples,fields)
        iter = data.BucketIterator(dataset, batch_size=64, train=False, shuffle=False, sort=False, device=device)

        decoded =self.model.decode(iter)
        segmented_sentence = self.BMSE2seg(sentences,decoded)
        return segmented_sentence
