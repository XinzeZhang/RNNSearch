from __future__ import print_function
import argparse
import time
import os
import sys
import subprocess
import tempfile

import torch.utils.data

from dataset import dataset, monoset
from util import convert_data, invert_vocab, load_vocab, convert_str,list_batch

import model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

parser = argparse.ArgumentParser(description='Testing Attention-based Neural Machine Translation Model')
# data
parser.add_argument('--src_vocab', default='./corpus/ldc_data/cn.voc3.pkl', type=str, help='source vocabulary')
parser.add_argument('--trg_vocab', default='./corpus/ldc_data/en.voc3.pkl', type=str, help='target vocabulary')
parser.add_argument('--src_max_len', type=int, default=50, help='maximum length of source')
parser.add_argument('--trg_max_len', type=int, default=50, help='maximum length of target')
parser.add_argument('--test_src', default='corpus/ldc/nist08/nist08.cn', type=str, help='source for testing')
parser.add_argument('--test_trg', default=['corpus/ldc/nist08/nist08.en0'], type=str, nargs='+', help='reference for testing')
parser.add_argument('--eval_script',default='scripts/validate.sh', type=str, help='script for validation')
# model
parser.add_argument('--model', default='RNNSearch', type=str, help='name of model')
parser.add_argument('--fname', default='cn-en/best.pt',type=str, help='name of checkpoint')
parser.add_argument('--bname', default='en-cn/best.pt',type=str, help='name of checkpoint')

parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to checkpoint')
parser.add_argument('--save', type=str, default='./generation/', help='path to save generated sequence')
# GPU
parser.add_argument('--cuda', default=True, help='use cuda')
# parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--verbose', default=False, help='use cuda')
#---------------------------------------------------------------------------------
parser.add_argument('--enc_ninp', type=int, default=620, help='size of source word embedding')
parser.add_argument('--dec_ninp', type=int, default=620, help='size of target word embedding')
parser.add_argument('--enc_nhid', type=int, default=1000, help='number of source hidden layer')
parser.add_argument('--dec_nhid', type=int, default=1000, help='number of target hidden layer')
parser.add_argument('--dec_natt', type=int, default=1000, help='number of target attention layer')
parser.add_argument('--nreadout', type=int, default=620, help='number of maxout layer')
parser.add_argument('--enc_emb_dropout', type=float, default=0.3, help='dropout rate for encoder embedding')
parser.add_argument('--dec_emb_dropout', type=float, default=0.3, help='dropout rate for decoder embedding')
parser.add_argument('--enc_hid_dropout', type=float, default=0.3, help='dropout rate for encoder hidden state')
parser.add_argument('--readout_dropout', type=float, default=0.3, help='dropout rate for encoder hidden state')
# search
parser.add_argument('--beam_size', type=int, default=10, help='size of beam')
# bookkeeping
parser.add_argument('--seed', type=int, default=123, help='random number seed')
# Misc
parser.add_argument('--info', type=str, help='info of the model')

opt = parser.parse_args()

# set the random seed manually
torch.manual_seed(opt.seed)

opt.cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device('cuda' if opt.cuda else 'cpu')

# load vocabulary for source and target
src_vocab, trg_vocab = {}, {}
src_vocab['stoi'] = load_vocab(opt.src_vocab)
trg_vocab['stoi'] = load_vocab(opt.trg_vocab)
src_vocab['itos'] = invert_vocab(src_vocab['stoi'])
trg_vocab['itos'] = invert_vocab(trg_vocab['stoi'])
UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
opt.enc_pad = src_vocab['stoi'][PAD]
opt.dec_sos = trg_vocab['stoi'][SOS]
opt.dec_eos = trg_vocab['stoi'][EOS]
opt.dec_pad = trg_vocab['stoi'][PAD]
opt.enc_ntok = len(src_vocab['stoi'])
opt.dec_ntok = len(trg_vocab['stoi'])

# load dataset for testing
forward_dataset = dataset(opt.test_src, opt.test_trg)
forward_iter = torch.utils.data.DataLoader(forward_dataset, 1, shuffle=False, collate_fn=lambda x: zip(*x))

back_dataset = dataset(opt.test_trg, opt.test_src)
back_iter = torch.utils.data.DataLoader(back_dataset, 1, shuffle=False, collate_fn=lambda x: zip(*x))

# create the forward and back translation models
fmodel = getattr(model, opt.model)(opt).to(device)
bmodel = getattr(model, opt.model)(opt).to(device)

f_state_dict = torch.load(os.path.join(opt.checkpoint, opt.fname))
fmodel.load_state_dict(f_state_dict)
fmodel.eval()

b_state_dict = torch.load(os.path.join(opt.checkpoint, opt.bname))
bmodel.load_state_dict(b_state_dict)
bmodel.eval()

def bleu_script(f):
    ref_stem = opt.test_src
    cmd = '{eval_script} {refs} {hyp}'.format(eval_script=opt.eval_script, refs=ref_stem, hyp=f)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        sys.exit(1)
    bleu = float(out)
    return bleu


hyp_list = []
ref_list = []
start_time = time.time()
for ix, batch in enumerate(forward_iter, start=1):
    batch = list_batch(batch)
    src_raw = batch[0]
    trg_raw = batch[1:]
    src, src_mask = convert_data(src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS)
    with torch.no_grad():
        output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
        best_hyp, best_score = output[0]
        best_hyp = convert_str([best_hyp], trg_vocab)
        hyp_list.append(best_hyp[0])
        ref = list(map(lambda x: x[0], trg_raw))
        ref_list.append(ref)
    if opt.verbose:
        print(ix, len(forward_iter), 100. * ix / len(forward_iter))
elapsed = time.time() - start_time
bleu1 = corpus_bleu(ref_list, hyp_list, smoothing_function=SmoothingFunction().method1)
hyp_list = list(map(lambda x: ' '.join(x), hyp_list))
p_tmp = tempfile.mktemp()
f_tmp = open(p_tmp, 'w')
f_tmp.write('\n'.join(hyp_list))
f_tmp.close()
bleu2 = bleu_script(p_tmp)
print('BLEU score for model {} is {}/{}, {}'.format(opt.fname, bleu1, bleu2, elapsed))