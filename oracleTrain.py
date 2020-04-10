import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    description='Preprocess of LDC corpus')
parser.add_argument('-checkpoint', type=str,
                    default='./RNNsearch/checkpoint/', metavar='S')
parser.add_argument('-pair', type=str, default='en-cn', metavar='S')
parser.add_argument('-len', type=int, default=80, metavar='N')
parser.add_argument('-name', type=str, default='best.pt', metavar='S')
parser.add_argument('-token', type=str, default='word', metavar='S')
parser.add_argument('-gpuid', type=int,default=0,metavar='N')

if __name__ == "__main__":
    opt = parser.parse_args()

    pair = opt.pair
    max_len = opt.len
    token = opt.token
    name = opt.name
    model_folder = opt.checkpoint + pair + '/'

    gpuid = opt.gpuid

    src = pair.split('-')[0] # default en
    tgt = pair.split('-')[1] # default cn

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    src_vocab,trg_vocab,train_src,train_trg,valid_src,valid_trg = None,None,None,None,None,None

    cmd = None
    if token == 'word':
        src_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(src)
        trg_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(tgt)
        train_src = './corpus/nist_para/nist.cn-en.{}'.format(src)
        train_trg = './corpus/nist_para/nist.cn-en.{}'.format(tgt)
        valid_src = './corpus/ldc_data/nist02/nist02.clean.{}'.format(src)
        valid_trg = './corpus/ldc_data/nist02/nist02.clean.{}'.format(tgt)

        if 'en' in valid_src:
            valid_src+='0'
        if 'en' in valid_trg:
            valid_trg+='0'

        cmd = 'python train.py \
            --src_vocab {} --trg_vocab {} \
            --train_src {} --train_trg {} \
            --valid_src {} \
            --valid_trg {} \
            --checkpoint {} \
            --name {} \
            --epoch_best \
            --local_rank {}\
            '.format(src_vocab,trg_vocab,train_src,train_trg,valid_src,valid_trg,model_folder,name,gpuid)

    print(cmd)
    os.system(cmd)