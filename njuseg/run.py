from segmenter import Segmenter
import argparse
from model.yaml_config import parse_options

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config_file','-c',default=None,type=str)
    args.add_argument('--embedding_dim',default=100,type=int)
    args.add_argument('--hidden_dim',type=int,default=100)
    args.add_argument('--clf_hidden_dim',type=int,default=100)
    args.add_argument('--epoch',type=int,default=20)
    args.add_argument('--batch_size',type=int,default=32)
    args.add_argument('--lstm_layer',type=int,default=2)
    args.add_argument('--dropout',type=float,default=0.5)
    args.add_argument('--model_pth',type=str)
    args.add_argument('--env','-e',default='train',type=str)
    args.add_argument('--test_target',type=str)
    args.add_argument('--data_dir',type=str)

    options = parse_options(args)

    if options.train:
        Segmenter.train(options)
    elif options.test:
        fscore = Segmenter.test(options.model_pth)
        print('Test fscore :{}'.format(fscore))
    elif options.seg:
        Segmenter.seg_corpus(options.model_pth,options.raw_corpus,options.output)
