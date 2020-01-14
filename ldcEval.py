import os
from datetime import datetime
import argparse

from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction, sentence_bleu

import sys
import subprocess
import tempfile

parser = argparse.ArgumentParser(
    description='Preprocess of LDC corpus')
parser.add_argument('-test', type=str,
                    default='nist03', metavar='S')
parser.add_argument('-pair', type=str, default='cn-en', metavar='S')
parser.add_argument('-len', type=int, default=80, metavar='N')
parser.add_argument('-gpuid', type=int, default=0, metavar='N')
parser.add_argument('-token', type=str, default='word', metavar='S')


def bleu_script(r,p):
    cmd = '{eval_script} {refs} {hyp}'.format(eval_script='scripts/validate.sh', refs=r, hyp=p)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        sys.exit(1)
    bleu = float(out)
    return bleu

def files2eval(r, p):
    refs = []
    with open(r, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            _l = list(l.split(' '))
            temp = []
            temp.append(_l)
            refs.append(temp)
    
    preds = []
    with open(p, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            _l = list(l.split(' '))
            preds.append(_l)

    assert len(refs) == len(preds)

    bleu1 = corpus_bleu(refs,preds, smoothing_function=SmoothingFunction().method1)

    sbleu=0
    for ref, pred in zip(refs,preds):
        sbleu += sentence_bleu(ref,pred,smoothing_function=SmoothingFunction().method1)
    bleu2 = sbleu / len(refs)
    return 100*bleu1, 100*bleu2




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

    for nist in testsets:
        ref_path1 = 'corpus/ldc_data/'+nist+'/'+nist+'.clean.pkuseg.cn'
        ref_path2 = 'corpus/ldc_data/'+nist+'/'+nist+'.clean.jieba.cn'
        ref_path3 = 'corpus/ldc_data/'+nist+'/'+nist+'.clean.cn'
        hyp_path = 'generation/baidu/'+nist+'/'+nist+'.baidu.pkuseg.cn'
                   
        
        bleu0 = bleu_script(ref_path1+' '+ ref_path2 + ' ' + ref_path3,hyp_path)
        print('BLEU score of baidu for the {} is {:.2f}'.format(nist,bleu0))
        # bleu1, bleu2 = files2eval(ref_path1,hyp_path)
        # print('BLEU score of baidu for the {} is {:.2f}/{:.2f}/{:.2f}'.format(nist,bleu0,bleu1,bleu2))


    # if token == 'word':
    #     src_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(src)
    #     trg_vocab = './corpus/ldc_data/{}.voc3.pkl'.format(tgt)

    #     for nist in testsets:
    #         cmd = 'bash sh-test-{}.sh {} {} {} {}'.format(pair, gpuid, nist, src_vocab, trg_vocab)
    #         print(cmd)

    #         os.system(cmd)
    