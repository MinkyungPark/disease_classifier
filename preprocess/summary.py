import pickle

# Dictionary LOAD
with open('data/word_index_v3.pickle', 'rb') as fr:
    word_index = pickle.load(fr)

with open('data/index_word_v3.pickle', 'rb') as fr:
    index_word = pickle.load(fr)

with open('data/category_to_idx_v3.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('data/idx_to_category_v3.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)

import gensim.models as g
import numpy as np

w2v_model = g.Doc2Vec.load('data/nin_w2v_20200406.model')

vocab = list(w2v_model.wv.vocab)
vector = w2v_model[vocab]

max_words = len(vocab)
embedding_dim = 256

embedding_index = {} # embedding_index = {단어:[단어벡터], ...}
for i in range(len(vocab)):
    embedding_index.setdefault(vocab[i], list(vector[i]))

embedding_matrix = np.zeros((max_words, embedding_dim)) # 임베딩 층에 주입.

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 임베딩 인덱스에 없는 단어는 0
            embedding_matrix[i] = embedding_vector
            # word_index의 index는 1부터 시작



# %% 5 MODEL
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.models import Model
from keras.layers import Dense, Bidirectional, Dropout, LSTM, Input, GlobalMaxPool1D
from keras.layers.embeddings import Embedding


model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=512, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=512, weights=[embedding_matrix], trainable=False))
# model.add(Dropout(0.2))
# model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2)) # 반으로
# # model.add(Dense(32, activation='relu'))
# model.add(LSTM(64))
# # model.add(Dropout(0.2))
# model.add(Dense(36, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# print(model.summary())