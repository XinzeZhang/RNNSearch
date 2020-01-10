import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    description='Preprocess of LDC corpus')
parser.add_argument('-test', type=str,
                    default='nist03', metavar='S')
parser.add_argument('-pair', type=str, default='cn-en', metavar='S')
parser.add_argument('-len', type=int, default=80, metavar='N')
parser.add_argument('-gpuid', type=int, default=0, metavar='N')
parser.add_argument('-token', type=str, default='word', metavar='S')

if __name__ == "__main__":
    opt = parser.parse_args()

    testset=opt.test
    pair = opt.pair
    max_len = opt.len
    token = opt.token
    gpuid = opt.gpuid

    src = pair.split('-')[0]
    tgt = pair.split('-')[1]

    testsets = ['nist02','nist03','nist04','nist05','nist06','nist08','avg',]

    if token == 'word':
        src_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(src)
        trg_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(tgt)

        for nist in testsets:
            cmd = 'bash sh-test-{}.sh {} {} {} {}'.format(pair, gpuid, nist, src_vocab, trg_vocab)
            print(cmd)

            os.system(cmd)