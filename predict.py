# %%
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)


import pickle

# Dictionary LOAD
with open('pickleFile/word_index.pickle', 'rb') as fr:
    word_index = pickle.load(fr)

with open('pickleFile/index_word.pickle', 'rb') as fr:
    index_word = pickle.load(fr)

with open('pickleFile/category_to_idx.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('pickleFile/idx_to_category.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)




#####
from keras.models import load_model
model = load_model('probability/cnn1d_model_8category_proba_0.h5')
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

sents_len = 500

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
        else: # 사전에 없으면 OOV 단어 추가
            sentence.extend([vocadic['OOV']])
        sentence += ' '

    return indexs, sentence



def get_category(speech):
    pre_speech = preprocess(speech)
    input_seq = txt_to_seq(pre_speech)
    input_seq = input_seq.reshape(1,500)
    result_idx = model.predict(input_seq)
    result_category = idx_to_txt(result_idx, idx_to_category)

    return pre_speech, result_category


if __name__ == "__main__":
    # test_df = pd.read_csv('new_model/for_test_001.csv', names=['sentences','category'], encoding='utf-8')
    # test_s = test_df['sentences']
    # test_c = test_df['category']

    # s = []
    # c = []
    # p = []
    # r = []

    # for i in range(len(test_s)):
    #     s.append(test_s[i])
    #     c.append(test_c[i])
    #     pre, result = get_category(test_s[i])
    #     p.append(pre)
    #     r.append(result)
    
    # for i in range(len(s)):
    #     print('RAW 문장 : ', s[i])
    #     print('-'*100)
    #     print('PREPROCESS 문장 : ', p[i])
    #     print('-'*100)
    #     print('TRUE : ', c[i])
    #     print('-'*100)
    #     print('PREDICT : ', r[i])
    #     print('*'*100)
    #     print('*'*100)


# 문장 입력해서 PREDICT
    # while(True):
    #     speech = input('증상 입력 >>>>> ')
    #     print('RAW 문장 : ', speech)
    #     print('-'*100)
    #     pre, result = get_category(speech)
    #     print('PREPROCESS 문장 : ', pre)
    #     print('-'*100)
    #     print('PREDICT : ', result)
    #     print('*'*100)


# PRDICT PROBABILITY
    while(True):
        speech = input('증상 입력 >>>>> ')
        print('RAW 문장 : ', speech)
        print('-'*100)
        pre_speech = preprocess(speech)
        print('PREPROCESS 문장 : ', pre_speech)
        print('-'*100)
        input_seq = txt_to_seq(pre_speech)
        input_seq = input_seq.reshape(1,500)
        result_idx = model.predict(input_seq)
        proba = model.predict_proba(input_seq)
        result_category = idx_to_txt(result_idx, idx_to_category)
        print('PROBABILITY : 1구취/2두통/3손발저림/4심계항진/5어지럼증/6요통/7잇몸염증/8하지마비')
        for i in proba:
            for j in i:
                print('%.8f'%j)
        print('-'*100)
        print('PREDICT : ', result_category)
        print('*'*100)

