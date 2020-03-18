from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

import pickle

with open('pickleFile/word_index.pickle', 'rb') as fr:
    word_index = pickle.load(fr)

with open('pickleFile/index_word.pickle', 'rb') as fr:
    index_word = pickle.load(fr)

with open('pickleFile/category_to_idx.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('pickleFile/idx_to_category.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)



df = pd.read_csv('probability/true_dataset.csv', encoding='utf-8')

sentences = list(df['sentences'].astype(str))
true = list(df['ture_value'].astype(int))

from keras.preprocessing.sequence import pad_sequences

sents_len = 500
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

y_data = np.asarray(true)
y_data = np_utils.to_categorical(y_data)


# %% 5 MODEL
from keras.models import load_model
model = load_model('probability/cnn1d_model_8category_proba_0.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    true = []
    pred = []
    
    for i in range(len(x_data)):
        true_idx = np.argmax(y_data[i], axis=-1)
        xx= x_data[i].reshape(1,500)
        tmp = model.predict(xx)
        pred_idx = np.argmax(tmp[0], axis=-1)

        true.append(true_idx)
        pred.append(pred_idx)

    print(confusion_matrix(true, pred))
    print(classification_report(true, pred, target_names=['구취','두통','손발저림','심계항진','어지럼증','요통','잇몸염증','하지마비']))

