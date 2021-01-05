import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('data3/total_36category.csv', names=['sentences', 'category'], encoding='utf-8')
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

# {0: '가래', 1: '가슴통증', 2: '고열', 3: '관절통', 4: '구취', 5: '구토', 6: '기침', 7: '다뇨', 8: '다식', 9: '다음', 10: '두통', 11: '반신마비', 12: '방사통', 13: '복부팽만', 14: '복시', 15: '복통', 16: '설사', 17: '소양감', 18: '소화불량', 19: '손발저림', 20: '시력감소', 21: '시야장애', 22: '식욕부진', 23: '심계항진', 24: '어지럼증', 25: '언어장애', 26: '연하곤란', 27: '오심', 28: '요통', 29: '운
# 동장애', 30: '잇몸염증', 31: '천명', 32: '체중감소', 33: '피로감', 34: '하지마비', 35: '호흡곤란'}

##### pickle로 저장
# word_index
# index_word
# category_to_idx
# idx_to_category


# SAVE
import pickle

with open('word_index_v3.pickle', 'wb') as fw:
    pickle.dump(word_index, fw)

with open('index_word_v3.pickle', 'wb') as fw:
    pickle.dump(index_word, fw)

with open('category_to_idx_v3.pickle', 'wb') as fw:
    pickle.dump(category_to_idx, fw)

with open('idx_to_category_v3.pickle', 'wb') as fw:
    pickle.dump(idx_to_category, fw)



# LOAD
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
