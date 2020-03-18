import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('data3/total_37category.csv', names=['sentences','category'], encoding='utf-8')

sents = list(df['sentences'].astype(str))
sents = sents[1:]

sentences = []

for line in sents:
    tmp = []
    for word in line.split():
        tmp.append(word)
    sentences.append(tmp)

# print(sentences[:10])

from gensim.models.word2vec import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(sentences, size=200, window=10, iter=300, workers=4, min_count=5, sg=1,)
model.init_sims(replace=True)

model.save('model/nin20200318_37category.model')

# # %%
# # 시각화...
# from sklearn.manifold import TSNE
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import gensim 
# import gensim.models as g
# import pandas as pd

# # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Malgun Gothic'

# model_name = 'new_model/nin20200303_long.model'
# model = g.Word2Vec.load(model_name)

# vocab = list(model.wv.vocab)
# X = model[vocab]

# print(len(X))
# print(X[0][:10])
# tsne = TSNE(n_components=2)

# # 500개의 단어에 대해서만 시각화
# X_tsne = tsne.fit_transform(X[:500,:])
# # X_tsne = tsne.fit_transform(X)

# df = pd.DataFrame(X_tsne, index=vocab[:500], columns=['x', 'y'])

# fig = plt.figure()
# fig.set_size_inches(60, 60)
# ax = fig.add_subplot(1, 1, 1)

# ax.scatter(df['x'], df['y'])

# for word, pos in df.iterrows():
#     ax.annotate(word, pos, fontsize=12)
# plt.show()


# # %%
# print(model.wv.similarity('재활','치료'))
# print(model.wv.similarity('증상','이상'))
# print(model.wv.similarity('체중','몸무게'))
# print()
# print(model.wv.most_similar('통증'))
# print()
# print(model.wv.most_similar('수술'))
# print()

# model.wv.doesnt_match('치료 재활 치료법 할인'.split()) # 혈압


