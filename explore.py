# %% 1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('data3/total_37category.csv', names=['sentences', 'category'], encoding='utf-8')
# data3 -> 1 data 1 question

print(df.info())

sentences = list(df['sentences'].astype(str))
category = list(df['category'].astype(str))

sentences = sentences[1:]
category = category[1:]

print(sentences[0])
print(category[0])

print(len(sentences))
print(len(category))



# %% 2
# tokenize 되어있는 데이터기 때문에 따로 konlpy 사용X

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # 빈도수에 기반한 사전
x_data = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index # word_idx 토큰의 갯수 22574개
index_word = {index+1: word for index, word in enumerate(word_index)}

print(len(word_index))



############### DATA EXPLORE ######################
# %% 문장 길이 확인
# token x char length o
print('문장 최대 토큰 갯수 : {}'.format(max([len(sentence) for sentence in x_data])))
print('문장 평균 토큰 갯수 : {}'.format(sum(map(len, x_data))/len(x_data)))

import matplotlib.pyplot as plt

plt.hist([len(s) for s in x_data], bins=50)
plt.xlabel('length of sentences')
plt.ylabel('number of sentences')
plt.show()

# %% 각 카테고리 당 빈도수 확인
unique_elements, counts_elements = np.unique(category, return_counts=True)
print("각 카테고리에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))



# %% 정수 인코딩, 패딩
word_index['OOV'] = 0
index_word[0] = 'OOV'

encoded=[]
for s in sentences:
    temp = []
    for w in s.split():
        try:
            temp.append(word_index[w])
        except KeyError:
            temp.append(word_index['OOV'])
    encoded.append(temp)


sents_len = 500 # 길이 500이상 그냥 자르기..
x_data = pad_sequences(encoded, maxlen=sents_len)

# y_data(category) 데이터 정수 인덱싱
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

category_set = list(set(category))
category_sort = sorted(category_set)
category_to_idx = {word: index for index, word in enumerate(category_sort)}
idx_to_category = {index: word for index, word in enumerate(category_sort)}

y_data = []

for word in category:
    y_data.append(category_to_idx[word])


from keras.utils import np_utils

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

print(x_data.shape) # , 500
print(y_data.shape) # , 8


# %% 3
# 데이터셋 섞기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

# %% 4 Embedding Layer에 주입할 w2v 모델 처리
# embedding_index = {단어:[단어벡터], ...}

import gensim.models as g

# w2v_model = g.Doc2Vec.load('model/nin20200222_all_.model')
w2v_model = g.Doc2Vec.load('model/nin20200318_37category.model')

vocab = list(w2v_model.wv.vocab)
vector = w2v_model[vocab]

max_words = len(vocab)
embedding_dim = 200

print(max_words) # 11529?????????????


# %%
embedding_index = {}
for i in range(len(vocab)):
    embedding_index.setdefault(vocab[i], list(vector[i]))


# (max_words, embedding_dim) 크기인 임베딩 행렬을 임베딩 층에 주입.

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 임베딩 인덱스에 없는 단어는 0
            embedding_matrix[i] = embedding_vector
            # word_index의 index는 1부터 시작

# print(embedding_matrix[0])
print(len(embedding_matrix))
