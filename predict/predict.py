# %%
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd
import sys

seed = 0
np.random.seed(seed)


import pickle

# Dictionary LOAD
with open('data/pickle/word_index_v3.pickle', 'rb') as fr:
    word_index = pickle.load(fr)

with open('data/pickle/index_word_v3.pickle', 'rb') as fr:
    index_word = pickle.load(fr)

with open('data/pickle/category_to_idx_v3.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('data/pickle/idx_to_category_v3.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)

# {0: '가래', 1: '가슴통증', 2: '고열', 3: '관절통', 4: '구취', 5: '구토', 6: '기침', 7: '다뇨', 8: '다식', 9: '다음', 10: '두통', 11: '반신마비', 12: '방사통', 13: '복부팽만', 14: '복시', 15: '복통', 16: '설사', 17: '소양감', 18: '소화불량', 19: '손발저림', 20: '시력감소', 21: '시야장애', 22: '식욕부진', 23: '심계항진', 24: '어지럼증', 25: '언어장애', 26: '연하곤란', 27: '오심', 28: '요통', 29: '운
# 동장애', 30: '잇몸염증', 31: '천명', 32: '체중감소', 33: '피로감', 34: '하지마비', 35: '호흡곤란'}




#####
from keras.models import load_model
model = load_model('android/h5/android_cnn_flatten_0.91097003_0.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from konlpy.tag import Kkma
from keras.preprocessing.sequence import pad_sequences

# 들어온 데이터 전처리

def preprocess(sents):
    kkma = Kkma()
    tagged = kkma.pos(sents)

    result = []
    for word, tag in tagged:        
        if tag in ['NNG']: #and len(word) > 1:
            result.append(word)
        if tag in ['VV', 'VA']:
            result.append(word + "다")

    result = ' '.join(result)

    return result

sents_len = 512

# 전처리된 데이터 sequences로 만들기
def txt_to_seq(sentence):
    encoded=[]
    tmp = []

    for w in sentence.split():
        try:
            tmp.append(word_index[w])
        except KeyError:
            tmp.append(word_index['OOV'])
    
    encoded.append(tmp)
    seq = pad_sequences(encoded, maxlen=sents_len)

    return seq[0]


def idx_to_txt(indexs, vocadic):
    indexs = np.argmax(indexs, axis=-1)
    sentence = ''

    for idx in indexs:
        if vocadic.get(idx) is not None:
            sentence += vocadic[idx]

    return indexs, sentence



def get_category(speech):
    pre_speech = preprocess(speech)
    input_seq = txt_to_seq(pre_speech)
    input_seq = input_seq.reshape(1,512)
    result_idx = model.predict(input_seq)
    proba = model.predict_proba(input_seq)
    result_category = idx_to_txt(result_idx, idx_to_category)

    return proba, pre_speech, result_category
    # return pre_speech, result_category



if __name__ == "__main__":
    # test_df = pd.read_csv('model/for_test_001.csv', names=['sentences','category'], encoding='utf-8')
    # test_s = test_df['sentences']
    # test_c = test_df['category']

    # s = []
    # c = []
    # p = []
    # r = []
    # pro = []

    # sys.stdout = open('result.txt','w')

    # for i in range(len(test_s)):
    #     s.append(test_s[i])
    #     c.append(test_c[i])
    #     proba, pre, result = get_category(test_s[i])
    #     p.append(pre)
    #     r.append(result)
    #     pro.append(proba)
    
    # for i in range(len(s)):
    #     print('RAW 문장 : ', s[i])
    #     print('-'*100)
    #     print('PREPROCESS 문장 : ', p[i])
    #     print('-'*100)
    #     print('TRUE : ', c[i])
    #     print('-'*100)
    #     print('PREDICT : ', r[i])
    #     print('*'*100)
    #     print('PROBABILITY : 0가래,1가슴통증,2고열,관절통,4구취,5구토,6기침,7다뇨,8다식,9다음,10두통,11반신마비,12방사통,13복부팽만,14복시,15복통,16설사,17소양감,18소화불량,19손발저림,20시력감소,21시야장애,22식욕부진,23심계항진,24어지럼증,25언어장애,26연하곤란,27오심,28요통,29운동장애,30잇몸염증,31천명,32체중감소,33피로감,34하지마비,35호흡곤란')
    #     for idx in pro[i]:
    #         for j in idx:
    #             print('%.8f'%j)
    #     print('*'*100)


# 문장 입력해서 PREDICT
    while(True):
        speech = input('증상 입력 >>>>> ')
        print('RAW 문장 : ', speech)
        print('-'*100)
        proba, pre, result = get_category(speech)
        print('PREPROCESS 문장 : ', pre)
        print('-'*100)
        print(proba[0][result[0][0]])
        print('PREDICT : ', result[1])
        print('*'*100)


# PRDICT PROBABILITY
    # while(True):
    #     speech = input('증상 입력 >>>>> ')
    #     print('RAW 문장 : ', speech)
    #     print('-'*100)
    #     pre_speech = preprocess(speech)
    #     print('PREPROCESS 문장 : ', pre_speech)
    #     print('-'*100)
    #     input_seq = txt_to_seq(pre_speech)
    #     input_seq = input_seq.reshape(1,500)
    #     result_idx = model.predict(input_seq)
    #     proba = model.predict_proba(input_seq)
    #     result_category = idx_to_txt(result_idx, idx_to_category)
    #     print('PROBABILITY : 1구취/2두통/3손발저림/4심계항진/5어지럼증/6요통/7잇몸염증/8하지마비')
    #     for i in proba:
    #         for j in i:
    #             print('%.8f'%j)
    #     print('-'*100)
    #     print('PREDICT : ', result_category)
    #     print('*'*100)

