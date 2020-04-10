import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    description='Preprocess of LDC corpus')
parser.add_argument('-checkpoint', type=str,
                    default='./RNNsearch/checkpoint/', metavar='S')
parser.add_argument('-pair', type=str, default='en-cn', metavar='S')
parser.add_argument('-len', type=int, default=80, metavar='N')
parser.add_argument('-gpuid', type=int, default=1, metavar='N')
parser.add_argument('-token', type=str, default='word', metavar='S')

if __name__ == "__main__":
    opt = parser.parse_args()

    pair = opt.pair
    max_len = opt.len
    token = opt.token
    gpuid = opt.gpuid
    model_folder = opt.checkpoint + pair + '/'

    src = pair.split('-')[0]
    tgt = pair.split('-')[1]

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

# python train.py \
# --src_vocab /path/to/cn.voc3.pkl --trg_vocab /path/to/en.voc3.pkl \
# --train_src corpus/train.cn-en.cn --train_trg corpus/train.cn-en.en \
# --valid_src corpus/nist02/nist02.cn \
# --valid_trg corpus/nist02/nist02.en0 corpus/nist02/nist02.en1 corpus/nist02/nist02.en2 corpus/nist02/nist02.en3 \
# --eval_script scripts/validate.sh \
# --model RNNSearch \
# --optim RMSprop \
# --batch_size 80 \
# --half_epoch \
# --cuda \
# --info RMSprop-half_epoch
    src_vocab,trg_vocab,train_src,train_trg,valid_src,valid_trg = None,None,None,None,None,None

    cmd = None
    if token == 'word':
        src_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(src)
        trg_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(tgt)
        train_src = './corpus/ldc/train.cn-en.{}'.format(src)
        train_trg = './corpus/ldc/train.cn-en.{}'.format(tgt)
        valid_src = './corpus/ldc/nist03/nist02.{}'.format(src)
        valid_trg = './corpus/ldc/nist03/nist02.{}'.format(tgt)

        if 'en' in valid_src:
            valid_src+='0'
        if 'en' in valid_trg:
            valid_trg+='0'

        cmd = 'python RNNsearch/train.py \
            --src_vocab {} --trg_vocab {} \
            --train_src {} --train_trg {} \
            --valid_src {} \
            --valid_trg {} \
            --checkpoint {} \
            --local_rank {} \
            '.format(src_vocab,trg_vocab,train_src,train_trg,valid_src,valid_trg,model_folder,gpuid)

    print(cmd)
    os.system(cmd)
