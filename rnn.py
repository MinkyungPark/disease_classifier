# %% 1
import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('data2/total.csv', names=['sentences', 'category'], encoding='utf-8')

sentences = list(df['sentences'])
category = list(df['category'])

sentences = sentences[1:]
category = category[1:]

# print(sentences)
# print(category)

# %% 2
# tokenize 되어있는 데이터기 때문에 따로 konlpy 사용X

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
x_data = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
# print(len(tokenizer.word_index)) # 토큰의 갯수 26809
# print(x_data[:10])

max_len = max([len(sentence) for sentence in x_data])
# print(max_len) # 267

x_data = pad_sequences(x_data, maxlen=max_len)


# y_data(category) 데이터 정수 인덱싱
category_sort = sorted(list(category))
category_set = list(set(category_sort))
category_to_idx = {word: index for index, word in enumerate(category_set)}

y_data = []

for word in category:
    y_data.append(category_to_idx[word])


from keras.utils import np_utils

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

# print(x_data[:10])
# print(y_data[:10])
print(x_data.shape)
# print(y_data.shape)

# %% 3
# 데이터셋 섞기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)
# X_train 978695개

import gensim.models as g

model = g.Doc2Vec.load('model//nin20200222_all_.model')

vocab = list(model.wv.vocab)
vector = model[vocab]

max_words = len(vocab)
embedding_dim = 100

print(len(vocab)) # 15652


# %% 4
# embedding_index = {단어:[단어벡터], ...}
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
            embedding_matrix[i-1] = embedding_vector
            # word_index의 index는 1부터 시작

# print(embedding_matrix[0])

# %% 5
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, SimpleRNN, RNN
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
model.add(Embedding(max_words, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False))
model.add(SimpleRNN(50))
model.add(Dropout(0.2))
model.add(Dense(y_data.shape[1], activation='softmax'))

print(model.summary())

# %% 6
### TRAIN ###

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
hist = model.fit(X_train, y_train, batch_size=267, epochs=5, callbacks=[early_stop], validation_split=0.2, verbose=1)

print('테스트 정확도 : %.4f' % (model.evaluate(X_test,y_test)[1]))

model.save_weights('model/cnn1d_model.h5')

# Epoch 5/5
# 782956/782956 [==============================] - 660s 843us/step - loss: 1.7141 - acc: 0.4019 - f1_m: 0.2629 - precision_m: 0.6818 - recall_m: 0.1681 - val_loss: 1.6667 - val_acc: 0.4139 - val_f1_m: 0.2783 - val_precision_m: 0.6973 - val_recall_m: 0.1742
# 244674/244674 [==============================] - 227s 930us/step
# 테스트 정확도 : 0.4128

# %% 7
# plot 그리기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
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

# ## evaluation loss, acc, f1, precision, recall ##
# 1.6711807415714315 0.4127696454524994 
# 0.2742612659931183 0.6938192248344421 0.17406335473060608

