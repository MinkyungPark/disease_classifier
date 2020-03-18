# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:46:24 2020

@author: Administrator
"""

import gensim
import os
import chardet
import codecs
import multiprocessing

file_name = "nin202002201654고혈압"

class SentenceReader:

    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        bytess = min(32, os.path.getsize(self.filepath))
        raw = open(self.filepath, 'rb').read(bytess)

        if raw.startswith(codecs.BOM_UTF8):
            encoding = 'utf-8-sig'
        else:
            result = chardet.detect(raw)
            encoding = result['encoding']
            
        for line in codecs.open(self.filepath, encoding=encoding):
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')            
            yield line.split(' ')


config = {
    'min_count': 5,  # 등장 횟수가 5 이하인 단어는 무시
    'size': 100,  # 300차원짜리 벡터스페이스에 embedding
    'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용한다
    'batch_words': 5000,  # 사전을 구축할때 한번에 읽을 단어 수
    'iter': 200,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수
    'window' : 10,
    'workers': multiprocessing.cpu_count(),
}

sentences_vocab = SentenceReader('data/' + file_name + '.txt')
sentences_train = SentenceReader('data/' + file_name + '.txt')

model = gensim.models.Word2Vec(**config)

model.build_vocab(sentences_vocab)
model.train(sentences_train, epochs=model.iter, total_examples=model.corpus_count)
model.save('model/' + file_name + '.model')

print(model)

## 테스트..
#model.wv.most_similar(positive=["요통"]) #test
#model.wv.similarity("허벅지", "종아리") #test


# 시각화...
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g
import pandas as pdmodel_name = 'new_model/nin20200228_long_.model'
model = g.Word2Vec.load(model_name)


print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# 500개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:500,:])
# X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab[:500], columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(60, 60)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=12)
plt.show()