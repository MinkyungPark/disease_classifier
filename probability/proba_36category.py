from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('probability2/false_dataset.csv', encoding='utf-8')
# data3 -> 1 data 1 question

sentences = list(df['sentences'].astype(str))
category = list(df['ture_value'].astype(str))

sentences = sentences[1:]
category = category[1:]



import pickle

with open('pickleFile/word_index_v3.pickle', 'rb') as fr:
    word_index = pickle.load(fr)

with open('pickleFile/index_word_v3.pickle', 'rb') as fr:
    index_word = pickle.load(fr)

with open('pickleFile/category_to_idx_v3.pickle', 'rb') as fr:
    category_to_idx = pickle.load(fr)

with open('pickleFile/idx_to_category_v3.pickle', 'rb') as fr:
    idx_to_category = pickle.load(fr)


encoded=[]
for s in sentences:
    temp = []
    for w in s.split():
        try:
            temp.append(word_index[w])
        except KeyError:
            temp.append(word_index['OOV'])
    encoded.append(temp)


from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

sents_len = 500 # 길이 500이상 그냥 자르기..
x_data = pad_sequences(encoded, maxlen=sents_len)


y_data = []
for word in category:
    y_data.append(category_to_idx[word])

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)



# 5 PRDICT PROBABILTY
from keras.models import load_model
model = load_model('model\cnn1d_model_36category_0.h5')
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
    result.to_csv('result.csv', index=False, encoding='utf-8')



    # print(confusion_matrix(true, pred))
    # print(classification_report(true, pred, ta''rget_names=['구취','두통','손발저림','심계항진','어지럼증','요통','잇몸염증','하지마비']))


# import pandas as pd

# df = pd.read_csv('result.csv', encoding='utf-8')

# result = list(df['result'].astype(str))

# false_df = pd.DataFrame()
# true_df = pd.DataFrame()

# for i in range(len(df)):
#     if result[i] == 'False':
#         false_df = false_df.append(df[i:i+1])
#     else:
#         true_df = true_df.append(df[i:i+1])


# false_df.to_csv('false_dataset.csv', index=False, encoding='utf-8')
# true_df.to_csv('true_dataset.csv', index=False, encoding='utf-8')