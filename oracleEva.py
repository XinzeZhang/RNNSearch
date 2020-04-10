import argparse
import os
from ldcEval import bleu_script,files2eval

parser = argparse.ArgumentParser(
    description='Preprocess of LDC corpus')
parser.add_argument('-test', type=str,
                    default='nist03', metavar='S')
parser.add_argument('-pair', type=str, default='cn-en', metavar='S')
parser.add_argument('-len', type=int, default=80, metavar='N')
parser.add_argument('-gpuid', type=int, default=0, metavar='N')
parser.add_argument('-token', type=str, default='word', metavar='S')

if __name__ == "__main__":
    nistSets = ['nist02']
    nistSets = ['nist02','nist03', 'nist04', 'nist05', 'nist06', 'nist08']
    pair = 'en-cn'
    src = pair.split('-')[0]
    tgt = pair.split('-')[1]
    state_name = 'obest'

    models=['rnn','bing','baidu','google']
    models=['rnn']
    for nist in nistSets:
        print(50*'-'+nist +'-'*50)
        for model in models:
            ref_path1 = 'corpus/ldc_data/'+nist+'/'+nist+'.clean.pkuseg.cn'
            # ref_path2 = 'corpus/ldc_data/'+nist+'/'+nist+'.clean.jieba.cn'
            # ref_path3 = 'corpus/ldc_data/'+nist+'/'+nist+'.clean.cn'
            hyp_path = 'generation/{}/'.format(model)+nist+'/'+nist+'.{}.pkuseg.cn'.format(model)
            if model =='rnn':
                hyp_path = 'generation/{}/'.format(model)+nist+'/'+nist+'.{}.pkuseg.{}.cn'.format(model,state_name)
                # hyp_path = 'generation/{}/'.format(model)+nist+'/'+nist+'.{}.raw.{}.cn'.format(model,state_name)
            # bleu0 = bleu_script(ref_path1+' '+ ref_path2 + ' ' + ref_path3,hyp_path)
            bleu0 = bleu_script(ref_path1,hyp_path)
            # print('BLEU score of baidu for the {} is {:.2f}'.format(nist,bleu0))
            bleu1, bleu2 = files2eval(ref_path1,hyp_path)
            print('BLEU score of {} for the {} is {:.2f}/{:.2f}'.format(model,nist,bleu0,bleu1))
