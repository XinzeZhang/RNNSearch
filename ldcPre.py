import jieba
import os
# jieba.enable_paddle()
# jieba.load_userdict("corpus/ldc/vocab.cn.txt")

def line2jieba(src_path, save_path):
    src = []
    char = []
    with open(src_path, encoding='utf-8') as f:
        for l in f:
            l = l.strip().replace(' ', '')
            charL = ' '.join(jieba.cut(l))
            charL = charL.strip()
            src.append(l)
            char.append(charL)

    assert len(src) == len(char)

    with open(save_path, mode='w') as f:
        for s in char:
            f.write(s.strip() + '\n')
    print('Save translation to '+save_path + ' Successfully!')   

def line2raw(src_path, save_path):
    src = []
    with open(src_path, encoding='utf-8') as f:
        for l in f:
            l = l.strip().replace(' ', '')
            src.append(l)

    with open(save_path, mode='w') as f:
        for s in src:
            f.write(s.strip() + '\n')
    print('Save translation to '+save_path + ' Successfully!')   
    

class docPre():
    def __init__(self, pair='cn-en', nist='nist02'):
        self.verbose = True
        self.nist = nist
        Path = 'corpus/ldc/'+self.nist+'/' + self.nist
        self.cn = Path + '.cn'
        self.en0 = Path + '.en0'
        self.en1 = Path+'.en1'
        self.en2 = Path + '.en2'
        self.en3 = Path+'.en3'

        iPath = 'corpus/ldc_data/'+self.nist+'/' + self.nist
        self.icn = iPath + '.clean.idx.cn'
        self.ien0 = iPath + '.clean.idx.en0'
        self.ien1 = iPath+'.clean.idx.en1'
        self.ien2 = iPath + '.clean.idx.en2'
        self.ien3 = iPath+'.clean.idx.en3'

    def pair_check(self, src, tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def loadFile(self, src_path, idx_path):
        src = []
        with open(src_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)

        idx = []
        with open(idx_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                idx.append(l)
        
        self.pair_check(src, idx)

        return src, idx
    
    def saveFile(self, src, save_path):
        with open(save_path, mode='w') as f:
            for s in src:
                f.write(s.strip() + '\n')
        print('Save translation to '+save_path + ' Successfully!')        

    def pair_pre(self):
        iPath = 'corpus/ldc_data/'+self.nist+'/' + self.nist
        self.pcn = iPath + '.clean.cn'
        self.pscn = iPath + '.clean.jieba.cn'
        self.pen0 = iPath + '.clean.en0'
        self.pen1 = iPath+'.clean.en1'
        self.pen2 = iPath + '.clean.en2'
        self.pen3 = iPath+'.clean.en3'

        lcn, licn = self.loadFile(self.cn, self.icn)
        len0, lien0 = self.loadFile(self.en0, self.ien0)
        len1, lien1 = self.loadFile(self.en1, self.ien1)
        len2, lien2 = self.loadFile(self.en2, self.ien2)
        len3, lien3 = self.loadFile(self.en3, self.ien3)

        tc, te0, te1,te2,te3=[],[],[],[],[]

        for lc, lic, le0, lie0, le1, lie1, le2, lie2, le3, lie3 in zip(lcn,licn,len0,lien0,len1,lien1,len2,lien2,len3,lien3):
            if lic == '2' and lie0 =='2' and lie1 =='2' and  lie2 =='2' and lie3 =='2':
                tc.append(lc)
                te0.append(le0)
                te1.append(le1)
                te2.append(le2)
                te3.append(le3)
        
        number = len(tc) 
        assert len(te0) == number
        assert len(te1) == number
        assert len(te2) == number
        assert len(te3) == number

        print('Raw lines : {}'.format(len(lcn)))
        print('Clean lines : {}'.format(number))
        self.saveFile(tc, self.pcn)
        self.saveFile(te0, self.pen0)
        self.saveFile(te1, self.pen1)
        self.saveFile(te2, self.pen2)
        self.saveFile(te3, self.pen3)

        line2jieba(self.pcn, self.pscn)



if __name__ == "__main__":
    nistSets = ['nist02','nist03', 'nist04','nist05', 'nist06', 'nist08']

    corpus_folder = 'generation/baidu_new/'
    # raw_folder = 'corpus/ldc/'
    for iNist in nistSets:
    #     pre = docPre(nist=iNist)
    #     pre.pair_pre()


        corpus = corpus_folder + iNist + '/' + iNist +'.baidu.raw.cn'
        
        jbPath = corpus_folder + iNist + '/' + iNist + '.baidu.jieba.cn'

        line2jieba(corpus, jbPath)


