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

df = pd.read_csv('total.csv', encoding='utf-8')
# data3 -> 1 data 1 question

sentences = list(df['sentences'].astype(str))
category = list(df['category'].astype(str))


# %% 2

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


from keras.preprocessing.sequence import pad_sequences

sents_len = 512
encoded=[]
for s in sentences:
    temp = []
    for w in s.split('/'):
        try:
            temp.append(word_index[w])
        except KeyError:
            temp.append(word_index['OOV'])
    encoded.append(temp)

x_data = pad_sequences(encoded, maxlen=sents_len)

from keras.utils import np_utils
from sklearn.model_selection import train_test_split

y_data = np.asarray(category)
y_data = np_utils.to_categorical(y_data)

# X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)
X_train = x_data
y_train = y_data

# %% 4
# Embedding Layer에 주입할 w2v 모델 처리
import gensim.models as g

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
from keras.models import Model
from keras.layers import Dense, Bidirectional, Dropout, LSTM, Input, GlobalMaxPool1D
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=sents_len, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_data.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# print(model.summary())
# %% 6 TRAIN

def idx_to_txt(indexs, vocadic):
    index = np.argmax(indexs, axis=-1)
    sentence = ''

    for idx in index:
        if vocadic.get(idx) is not None:
            sentence += vocadic[idx]
        else: # 사전에 없으면 OOV 단어 추가
            sentence.extend([vocadic['OOV']])
        sentence += ' '

    return index, sentence



from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=10, mode='auto')


for epoch in range(5):
    print('Total Epoch :', epoch+1)

    history = model.fit(X_train, y_train, batch_size=128, epochs=300, callbacks=[early_stop], verbose=1, shuffle=True) # validation_split=0.25

    print('ACC : ', history.history['acc'][-1])
    print('LOSS : ', history.history['loss'][-1])
    # print('테스트 정확도 : %.4f' % (model.evaluate(X_test,y_test)[1]))

    model.save('android_lstm_'+ str(history.history['acc'][-1]) + '_' + str(epoch) + '.h5')

    list_input = []
    list_true = []
    list_predict = []

    def test_predict(t):
        xx = X_train[t].reshape(1,X_train[t].shape[0])
        result_idx = model.predict(xx)
        idx, result_cate = idx_to_txt(result_idx, idx_to_category)
        true = np.argmax(y_train[t])

        input_sents = []
        for token in X_train[t]:
            if token == 0:
                pass
            else:
                input_sents.append(index_word[token])
        input_sents = '/'.join(input_sents)

        list_input.append(input_sents)
        list_true.append(true)
        list_predict.append(str(idx)+str(result_cate))


    test = [3, 10, 50, 500, 700, 1000, 1300, 1800, 3000, 10000, 20000]
    for t in test:
        test_predict(t)

    for i in range(len(list_input)):
        print('INPUT : ', list_input[i])
        print('TRUE : ', list_true[i])
        print('PREDICT : ', list_predict[i])
        print('*'*120)



# %% 6
# plot 그리기
# import matplotlib.pyplot as plt

# fig, loss_ax = plt.subplots()

# acc_ax = loss_ax.twinx()

# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_ylim([0.0, 3.0])

# acc_ax.plot(history.history['acc'], 'b', label='train acc')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylim([0.0, 1.0])

# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')

# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')

# plt.show()

# # 모델 평가
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print('## evaluation loss, acc ##')
# print(loss, accuracy)