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

df = pd.read_csv('true_dataset_total.csv', encoding='utf-8')
# data3 -> 1 data 1 question

sentences = list(df['sentences'].astype(str))
category = list(df['ture_value'].astype(str))



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
import pickle


with open('category_to_idx_v3.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('idx_to_category_v3.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)

y_data = np.asarray(category)
y_data = np_utils.to_categorical(y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)


# %% 4
# Embedding Layer에 주입할 w2v 모델 처리
import gensim.models as g

w2v_model = g.Doc2Vec.load('nin20200319_36category.model')

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



# %% 5 PRDICT PROBABILTY
from keras.models import load_model
model = load_model('bilstm/bilstm_model_36category_0.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    sent = []
    true = []
    prob = []
    pred = []
    res = []
    
    for i in range(len(x_data)):
        true_idx = np.argmax(y_data[i], axis=-1)
        xx= x_data[i].reshape(1,500)
        pro_tmp = model.predict_proba(xx)
        pre_tmp = model.predict(xx)
        pred_idx = np.argmax(pre_tmp[0], axis=-1)

        input_sents = []
        for token in x_data[i]:
            if token == 0:
                pass
            else:
                input_sents.append(index_word[token])
        input_sents = '/'.join(input_sents)

        sent.append(input_sents)

        true.append(true_idx)
        t = []
        for i in pro_tmp:
            for j in i:
                t.append(j)
        prob.append(t)
        pred.append(pred_idx)

        if true_idx == pred_idx:
            res.append('TRUE')
        else:
            res.append('FALSE')



    sent_col = pd.DataFrame(sent, columns=['sentences'])
    prob_col = pd.DataFrame(prob, columns=['0가래','1가슴통증','2고열','3관절통','4구취','5구토','6기침','7다뇨','8다식','9다음','10두통','11반신마비','12방사통','13복부팽만',\
          '14복시','15복통','16설사','17소양감','18소화불량','19손발저림','20시력감소','21시야장애','22식욕부진','23심계항진','24어지럼증','25언어장애',\
           '26연하곤란','27오심','28요통','29운동장애','30잇몸염증','31천명','32체중감소','33피로감','34하지마비','35호흡곤란'])
    true_col = pd.DataFrame(true, columns=['ture_value'])
    pred_col = pd.DataFrame(pred, columns=['predict_value'])
    res_col = pd.DataFrame(res, columns=['result'])

    result = pd.concat([sent_col, prob_col, true_col, pred_col, res_col], axis=1)
    result.to_csv('bilstm/result.csv', index=False, encoding='utf-8')



    # print(confusion_matrix(true, pred))
    # print(classification_report(true, pred, ta''rget_names=['구취','두통','손발저림','심계항진','어지럼증','요통','잇몸염증','하지마비']))





# %% concat
import pandas as pd

df = pd.read_csv('bilstm/result.csv', encoding='utf-8')

result = list(df['result'].astype(str))

false_df = pd.DataFrame()
true_df = pd.DataFrame()

for i in range(len(df)):
    if result[i] == 'False':
        false_df = false_df.append(df[i:i+1])
    else:
        true_df = true_df.append(df[i:i+1])


false_df.to_csv('bilstm/false_dataset.csv', index=False, encoding='utf-8')
true_df.to_csv('bilstm/true_dataset.csv', index=False, encoding='utf-8')