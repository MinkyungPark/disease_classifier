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

df = pd.read_csv('data3/total_11category.csv', names=['sentences', 'category'], encoding='utf-8')
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
word_index = tokenizer.word_index
print(len(word_index)) # 토큰의 갯수 21696
print(x_data[:10])

max_len = max([len(sentence) for sentence in x_data])
print(max_len) # 1187

sents_len = 400 # 400이상 그냥 자르기..


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


# %% 단어들의 빈도 수 확인
# word_idx 토큰의 갯수 21696개
# # 인덱스가 21696인 단어(빈도수 꼴등) '뾰족함'
for key, idx in word_index.items():
    if idx <= 10:
        print(key)


########################################################################################


# %% 빈도수 높은순으로 15000초과인 단어들 제거
vocab_size = 15000
words_frequency = [w for w,c in word_index.items() if c >= vocab_size + 1]

for w in words_frequency:
    del word_index[w]

word_index['OOV'] = len(word_index)+1

for key, idx in word_index.items():
    if idx >= 14999:
        print(key)


# %% 정수 인코딩, 패딩
encoded=[]
for s in sentences:
    temp = []
    for w in s.split():
        try:
            temp.append(word_index[w])
        except KeyError:
            temp.append(word_index['OOV'])
    encoded.append(temp)


x_data = pad_sequences(encoded, maxlen=sents_len)

# y_data(category) 데이터 정수 인덱싱
category_set = list(set(category))
category_sort = sorted(category_set)
category_to_idx = {word: index for index, word in enumerate(category_sort)}
category_to_idx['OOV'] = len(category_to_idx)
idx_to_category = {index: word for index, word in enumerate(category_sort)}

y_data = []

for word in category:
    y_data.append(category_to_idx[word])


from keras.utils import np_utils

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

# print(x_data[:10])
# print(y_data[:10])
print(x_data.shape) # 54812, 400
print(y_data.shape) # 54812, 10

print(category_to_idx)

# %% 3
# 데이터셋 섞기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)


# %% 4 Embedding Layer에 주입할 w2v 모델 처리
# embedding_index = {단어:[단어벡터], ...}

import gensim.models as g

# w2v_model = g.Doc2Vec.load('model/nin20200222_all_.model')
w2v_model = g.Doc2Vec.load('new_model/nin20200228_long_.model')

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

# %% 5
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras import backend as K

# F1 score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# MODEL
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=sents_len, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(filters=30, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(y_data.shape[1], activation='softmax'))

# %% 6
### TRAIN ###

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=10, mode='auto')

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m, precision_m, recall_m])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(X_train, y_train, epochs=200, callbacks=[early_stop], validation_split=0.2, verbose=1)

print('테스트 정확도 : %.4f' % (model.evaluate(X_test,y_test)[1]))

model.save('new_model/cnn1d_model_v2_epo200.h5') ############



# %%
# predict

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



def main():
    s = []
    seq = []
    ix = []
    p = []

    for i in [3, 500, 5000, 10000, 15000, 20000, 25000, 30000, 35079]:

        s.append(sentences[i])

        input_seq = txt_to_seq(sentences[i])
        input_seq = input_seq.reshape(1,sents_len)

        seq.append(input_seq)

        result_idx = model.predict(input_seq)

        idx, result_category = idx_to_txt(result_idx, idx_to_category)

        ix.append(idx)
        p.append(result_category)
    
    for i in range(len(s)):
        print('입력문장 : ', s[i])
        print('input seq : ', seq[i])
        print('result idx : ', ix[i])
        print('예측카테고리 : ', p[i])
        print('-'*100)


main()













# # %% 6
# # plot 그리기
# import matplotlib.pyplot as plt

# fig, loss_ax = plt.subplots()

# acc_ax = loss_ax.twinx()

# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# loss_ax.set_ylim([0.0, 3.0])

# acc_ax.plot(hist.history['acc'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylim([0.0, 1.0])

# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')

# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')

# plt.show()

# # 모델 평가
# loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
# print('## evaluation loss, acc, f1, precision, recall ##')
# print(loss, accuracy, f1_score, precision, recall)

