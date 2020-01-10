data_path=$PWD'/corpus/ldc/'
run_path=$PWD'/corpus/ldc_data/'
valid_folder='nist03/nist03.'

# for l in en cn; do for f in $data_path/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
# for l in en cn; do for f in $data_path/*.$l; do perl tools/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done

en_path=$data_path'train.cn-en.en.atok'
en_bpe_path=$run_path'train.en.bpe'

en_valid_path=$data_path$valid_folder'en3'
en_valid_bpe_path=$run_path$valid_folder'en3.bpe'

zh_path=$data_path'train.cn-en.cn.atok'
zh_bpe_path=$run_path'train.cn.bpe'

zh_valid_path=$data_path$valid_folder'cn'
zh_valid_bpe_path=$run_path$valid_folder'cn.bpe'

share_codes=$run_path'train.share.codes'
en_bpe_vocab=$run_path'train.en.bpe.vocab'
zh_bpe_vocab=$run_path'train.cn.bpe.vocab'

# pip install subword-nmt


# cat $zh_path $en_path | subword-nmt learn-bpe -s 60000 -o $share_codes
# subword-nmt apply-bpe -c $share_codes < $zh_path | subword-nmt get-vocab > $zh_bpe_vocab
# subword-nmt apply-bpe -c $share_codes < $en_path | subword-nmt get-vocab > $en_bpe_vocab

# subword-nmt learn-joint-bpe-and-vocab --input $zh_path $en_path -s 60000 -o $share_codes --write-vocabulary $zh_bpe_vocab $en_bpe_vocab

# subword-nmt apply-bpe -c $share_codes --vocabulary $en_bpe_vocab --vocabulary-threshold 50 < $en_path > $en_bpe_path

subword-nmt apply-bpe -c $share_codes --vocabulary $en_bpe_vocab --vocabulary-threshold 50 < $en_valid_path > $en_valid_bpe_path

# subword-nmt apply-bpe -c $share_codes --vocabulary $zh_bpe_vocab --vocabulary-threshold 50 < $zh_path > $zh_bpe_path

# subword-nmt apply-bpe -c $share_codes --vocabulary $zh_bpe_vocab --vocabulary-threshold 50 < $zh_valid_path > $zh_valid_bpe_path