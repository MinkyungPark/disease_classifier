import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('data3/total_8category.csv', names=['sentences', 'category'], encoding='utf-8')
# data3 -> 1 data 1 question

sentences = list(df['sentences'].astype(str))
category = list(df['category'].astype(str))

sentences = sentences[1:]
category = category[1:]


##### sentences dictionary
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # 빈도수에 기반한 사전
x_data = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index # word_idx 토큰의 갯수 22574개
index_word = {index+1: word for index, word in enumerate(word_index)}
word_index['OOV'] = 0
index_word[0] = 'OOV'


##### category dictionary
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

category_set = list(set(category))
category_sort = sorted(category_set)
category_to_idx = {word: index for index, word in enumerate(category_sort)}
idx_to_category = {index: word for index, word in enumerate(category_sort)}

# {'고혈압': 0, '구취': 1, '두통': 2, '디스크': 3, '손발저림': 4, '심계항진': 5, '어지럼증': 6, '요통': 7, '잇몸염증': 8, '치은염': 9, '하지마비': 10}
# {0: '구취', 1: '두통', 2: '손발저림', 3: '심계항진', 4: '어지럼증', 5: '요통', 6: '잇몸염증', 7: '하지마비'}


##### pickle로 저장
word_index
index_word
category_to_idx
idx_to_category


import pickle

with open('word_index.pickle', 'wb') as fw:
    pickle.dump(word_index, fw)

with open('index_word.pickle', 'wb') as fw:
    pickle.dump(index_word, fw)

with open('category_to_idx.pickle', 'wb') as fw:
    pickle.dump(category_to_idx, fw)

with open('idx_to_category.pickle', 'wb') as fw:
    pickle.dump(idx_to_category, fw)




# with open('word_index.pickle', 'rb') as fr:
#     word_index = pickle.load(fr)

# with open('index_word.pickle', 'rb') as fr:
#     index_word = pickle.load(fr)

# with open('category_to_idx.pickle', 'rb') as fr:
#     category_to_idx = pickle.load(fr)

# with open('idx_to_category.pickle', 'rb') as fr:
#     idx_to_category = pickle.load(fr)

# print(word_index['OOV'])
# print(index_word[0])
# print(category_to_idx['두통'])
# print(idx_to_category[1])
