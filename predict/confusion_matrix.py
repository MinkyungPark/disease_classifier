from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys
import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

import pickle

with open('data/word_index_v3.pickle', 'rb') as fr:
    word_index = pickle.load(fr)

with open('data/index_word_v3.pickle', 'rb') as fr:
    index_word = pickle.load(fr)

with open('data/category_to_idx_v3.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('data/idx_to_category_v3.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)



df = pd.read_csv('data/test_total_pos.csv', encoding='utf-8')

sentences = list(df['sentences'].astype(str))
category = list(df['category'].astype(str))


from keras.preprocessing.sequence import pad_sequences

sents_len = 512
encoded=[]
for s in sentences:
    temp = []
    for w in s.split(' '):
        try:
            temp.append(word_index[w])
        except KeyError:
            temp.append(word_index['OOV'])
    encoded.append(temp)

x_data = pad_sequences(encoded, maxlen=sents_len)


from keras.utils import np_utils

y_data = []
for word in category:
    y_data.append(category_to_idx[word])

y_data = np.asarray(y_data)

# y_data = np.asarray(category)
y_data = np_utils.to_categorical(y_data)



# %% 5 MODEL
from keras.models import load_model
model = load_model('bilstm_model64_total_true_0.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    true = []
    pred = []
    
    for i in range(len(x_data)):
        true_idx = np.argmax(y_data[i], axis=-1)
        xx= x_data[i].reshape(1,512)
        tmp = model.predict(xx)
        pred_idx = np.argmax(tmp[0], axis=-1)

        true.append(true_idx)
        pred.append(pred_idx)

    target_names=['0가래','1가슴통증','2고열','3관절통','4구취','5구토','6기침','7다뇨','8다식','9다음','10두통','11반신마비','12방사통','13복부팽만',\
            '14복시','15복통','16설사','17소양감','18소화불량','19손발저림','20시력감소','21시야장애','22식욕부진','23심계항진','24어지럼증','25언어장애',\
            '26연하곤란','27오심','28요통','29운동장애','30잇몸염증','31천명','32체중감소','33피로감','34하지마비','35호흡곤란']

    # print(confusion_matrix(true, pred))
    # print(classification_report(true, pred, target_names=target_names))

    cm = confusion_matrix(true, pred)
    cm_pd = pd.DataFrame(cm)
    cm_pd.to_csv('confusion_matrix.csv')
    with open('confusion_matrix.pickle', 'wb') as fw:
        pickle.dump(cm, fw)


    cr = classification_report(true, pred) # , target_names=target_names
    print(cr)
    with open('classification_repor.pickle', 'wb') as fw:
        pickle.dump(cr, fw)
    cr_pd = pd.DataFrame(cr)
    cr_pd.to_csv('classification_report.csv')

    # with open('confusion_matrix.pickle', 'rb') as fr:
    #     cm = pickle.load(fr)

    # cm_pd = pd.DataFrame(cm)
    # cm_pd.to_csv('confusion_matrix.csv')