from __future__ import print_function
import argparse
import time
import os
import sys
import subprocess
import tempfile

import torch.utils.data

from RNNsearch.dataset import dataset
from RNNsearch.util import convert_data, invert_vocab, load_vocab, convert_str, list_batch, listToString, line2seg
from RNNsearch import model


class rnn_translator():
    def __init__(self, src_lang, tgt_lang, state_name='best'):
        if tgt_lang == 'zh':
            tgt_lang = 'cn'
        if src_lang == 'zh':
            src_lang = 'cn'

        self.src = src_lang
        self.src_vocab_path = './corpus/ldc_data/{}.voc3.pkl'.format(src_lang)
        self.src_max_len = 50
        self.trg = tgt_lang
        self.trg_vocab_path = './corpus/ldc_data/{}.voc3.pkl'.format(tgt_lang)
        self.trg_max_len = 50
        self.model_name = 'RNNSearch'

        self.enc_ninp = 620
        self.dec_ninp = 620
        self.enc_nhid = 1000
        self.dec_nhid = 1000
        self.dec_natt = 1000
        self.nreadout = 620
        self.enc_emb_dropout = 0.3
        self.dec_emb_dropout = 0.3
        self.enc_hid_dropout = 0.3
        self.readout_dropout = 0.3

        self.beam_size = 10
        self.seed = 123
        self.checkpoint = './RNNSearch/checkpoint/'
        self.model_direction = src_lang + '-' + \
            tgt_lang + '/{}.pt'.format(state_name)

        self.cuda = False
        self.verbose = False

        torch.manual_seed(self.seed)

        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.src_vocab, self.trg_vocab = {}, {}
        self.src_vocab['stoi'] = load_vocab(self.src_vocab_path)
        self.trg_vocab['stoi'] = load_vocab(self.trg_vocab_path)
        self.src_vocab['itos'] = invert_vocab(self.src_vocab['stoi'])
        self.trg_vocab['itos'] = invert_vocab(self.trg_vocab['stoi'])

        self.UNK = '<unk>'
        self.SOS = '<sos>'
        self.EOS = '<eos>'
        self.PAD = '<pad>'

        self.enc_pad = self.src_vocab['stoi'][self.PAD]
        self.dec_sos = self.trg_vocab['stoi'][self.SOS]
        self.dec_eos = self.trg_vocab['stoi'][self.EOS]
        self.dec_pad = self.trg_vocab['stoi'][self.PAD]
        self.enc_ntok = len(self.src_vocab['stoi'])
        self.dec_ntok = len(self.trg_vocab['stoi'])

        self.model = getattr(model, self.model_name)(self).to(self.device)

        state_path = os.path.join(self.checkpoint, self.model_direction)
        state_dict = torch.load(state_path, map_location=self.device)
        print('Loading model state in {}'.format(state_path))

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def translate(self, src):
        pred = None
        s = src.strip()
        _s = list(s.split(' '))
        src_raw = []
        src_raw.append(_s)
        _src, src_mask = convert_data(
            src_raw, self.src_vocab, self.device, True, self.UNK, self.PAD, self.SOS, self.EOS)
        with torch.no_grad():
            output = self.model.beamsearch(
                _src, src_mask, self.beam_size, normalize=True)
            best_hyp, best_score = output[0]
            best_hyp = convert_str([best_hyp], self.trg_vocab)
            pred = best_hyp[0]
            pred = listToString(pred)

        assert pred != None

        return src, pred

    def pair_check(self, src, tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def dco_translate(self, src_path, save_path):
        src = []
        with open(src_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)

        preds = []
        count = 0
        for l in src:
            # time.sleep(0.101)
            _, pred = self.translate(l)
            pred = pred.strip()
            count += 1
            print('-'*30)
            print("Sentences: " + str(count))
            print('Src: '+_)
            print('====>')
            print('Pred: '+pred)
            preds.append(pred)

        self.pair_check(src, preds)

        with open(save_path, mode='w') as f:
            for s in preds:
                f.write(s.strip() + '\n')
        print('Save translation to '+save_path + ' Successfully!')


if __name__ == "__main__":
    # srcLang,tgtLang = 'cn', 'en'
    srcLang, tgtLang = 'en', 'cn'
    state_name = 'obest'

    rnn = rnn_translator(
        src_lang=srcLang, tgt_lang=tgtLang, state_name=state_name)
    # _, pred =rnn.translate('export of high-tech products in guangdong in first two months this year reached 3.76 billion us dollars')
    # print(pred)

    # nistSets = ['nist02']
    nistSets = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08']

    for nist in nistSets:
        src_path = 'corpus/ldc_data/' + nist + \
            '/' + nist + '.clean.{}'.format(srcLang)
        if srcLang == 'en':
            src_path += '0'
        save_path = 'generation/rnn/' + nist + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        seg_path = save_path + nist + \
            '.rnn.pkuseg.{}.{}'.format(state_name, tgtLang)
        save_path += nist+'.rnn.raw.{}.{}'.format(state_name, tgtLang)

        rnn.dco_translate(src_path, save_path)

        line2seg(save_path, seg_path)
