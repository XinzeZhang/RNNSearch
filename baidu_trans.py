#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import time
import os

class baidu_translator:
    def __init__(self, src_lang, tgt_lang):
        self.fromlang =src_lang
        self.tolang = tgt_lang
        self.appid = os.getenv('BAIDU_APPID')# 填写你的appid
        self.secretKey = os.getenv('BAIDU_KEY') # 填写你的密钥
        self.httpClient = None
        self.myurl = '/api/trans/vip/translate'

    def translate(self,src):
        q = src.strip()
        salt = random.randint(32768, 65536)
        sign = self.appid + q + str(salt) + self.secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        qurl = self.myurl + '?appid=' + self.appid + '&q=' + urllib.parse.quote(q) + '&from=' + self.fromlang + '&to=' + self.tolang + '&salt=' + str(salt) + '&sign=' + sign

        source = None
        translation = None
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', qurl)

            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)
            source = result.get('trans_result')[0].get('src')
            translation = result.get('trans_result')[0].get('dst')

        except Exception as e:
            print (e)
        finally:
            if httpClient:
                httpClient.close()
            
        return source, translation
    
    def pair_check(self,src,tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def dco_translate(self, src_path, save_path):
        src = []
        with open(src_path,encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)
        
        preds = []
        count = 0
        for l in src:
            time.sleep(1.001)
            _,pred = self.translate(l)
            pred=pred.strip()
            count += 1
            print('-'*30)
            print("Sentences: " + str(count))
            print('Src: '+_)
            print('====>')
            print('Pred: '+pred)
            preds.append(pred)
        
        self.pair_check(src,preds)

        with open(save_path, mode='w') as f:
            for s in preds:
                f.write(s.strip() + '\n')
        print('Save translation to '+save_path+ ' Successfully!')


if __name__ == "__main__":
    baidu = baidu_translator(src_lang='auto',tgt_lang='zh')
    # src_path = 'nist03.en0'
    src_path = 'dev.en'
    save_path = 'pred.cn'
    testsets = ['nist02','nist03','nist04','nist05','nist06','nist08','avg',]

    for nist in testsets:
        src_path = 'corpus/ldc/'+nist+'/' +nist + '.en0'
        save_path = 'generation/baidu/'+nist +'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path+=nist+'.raw.cn'
        baidu.dco_translate(src_path,save_path)