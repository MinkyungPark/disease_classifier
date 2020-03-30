# %% 1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd

seed = 30
np.random.seed(seed)

df = pd.read_csv('data3/total_36category.csv', names=['sentences', 'category'], encoding='utf-8')
# data3 -> 1 data 1 question

sentences = list(df['sentences'].astype(str))
category = list(df['category'].astype(str))

sentences = sentences[1:]
category = category[1:]



# %% 2
# tokenize 되어있는 데이터기 때문에 따로 konlpy 사용X

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # 빈도수에 기반한 사전
x_data = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index # word_idx 토큰의 갯수 22574개
index_word = {index+1: word for index, word in enumerate(word_index)}
word_index['OOV'] = 0
index_word[0] = 'OOV'

# print(len(word_index)) # 22575

# 인덱스가 21696인 단어(빈도수21696번째) '뾰족함'
# 빈도수 높은순으로 15000초과인 단어들 제거
# vocab_size = 15000
# words_frequency = [w for w,c in word_index.items() if c >= vocab_size + 1]
# for w in words_frequency:
#     del word_index[w]


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


# %% 3 y_data(category) 데이터 정수 인덱싱
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

category_set = list(set(category))
category_sort = sorted(category_set)
category_to_idx = {word: index for index, word in enumerate(category_sort)}
idx_to_category = {index: word for index, word in enumerate(category_sort)}

print(idx_to_category)
print(category_to_idx)
# {0: '구취', 1: '두통', 2: '손발저림', 3: '심계항진', 4: '어지럼증', 5: '요통', 6: '잇몸염증', 7: '하지마비'}

y_data = []
for word in category:
    y_data.append(category_to_idx[word])

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, test_size=0.2)

print(x_data.shape)
print(y_data.shape)

# %% 4
# Embedding Layer에 주입할 w2v 모델 처리
import gensim.models as g

w2v_model = g.Doc2Vec.load('model/nin20200319_36category.model')

vocab = list(w2v_model.wv.vocab)
vector = w2v_model[vocab]

max_words = len(vocab)
embedding_dim = 200

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


print(max_words) # 18041 w2v num of vacab
print(word_index['하다'])
print(embedding_index['하다'])
print(embedding_matrix[1])

# %% 5 MODEL
from keras.models import Sequential
# from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.models import Model
from keras.layers import Dense, Bidirectional, Dropout, LSTM, Input, GlobalMaxPool1D
from keras.layers.embeddings import Embedding

# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=sents_len, weights=[embedding_matrix], trainable=False))
# model.add(Conv1D(filters=20, kernel_size=5, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2)) # 반으로
# model.add(LSTM(50))
# model.add(Dropout(0.2))
# # model.add(Dense(30, activation='sigmoid'))
# model.add(Dense(y_data.shape[1], activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Bidirectional LSTM
# def model():
#     inp = Input(shape = (max_words, ))
#     layer = Embedding(max_words, embedding_dim, input_length=sents_len, weights=[embedding_matrix], trainable=False)(inp)
#     layer = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.2))(layer)
#     layer = GlobalMaxPool1D()(layer)
#     layer = Dropout(0.2)(layer)
#     layer = Dense(50, activation='relu')(layer)
#     layer = Dropout(0.2)(layer)
#     layer = Dense(y_data.shape[1], activation='softmax')(layer)
#     model = Model(inputs=inp, outputs=layer)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     return model

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=sents_len, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.2)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_data.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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


for epoch in range(1):
    print('Total Epoch :', epoch+1)

    history = model.fit(X_train, y_train, batch_size=128, epochs=300, callbacks=[early_stop], verbose=1, validation_split=0.25)

    print('ACC : ', history.history['acc'][-1])
    print('LOSS : ', history.history['loss'][-1]) # train 49698
    print('테스트 정확도 : %.4f' % (model.evaluate(X_test,y_test)[1])) # 12425

    model.save('model/cnn1d_model_36category_'+ str(epoch) + '.h5') #################

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
        input_sents = '//'.join(input_sents)

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
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 평가
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print('## evaluation loss, acc, f1, precision, recall ##')
print(loss, accuracy, f1_score, precision, recall)