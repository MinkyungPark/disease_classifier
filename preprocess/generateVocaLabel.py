import sys

sys.stdout = open('label.txt', 'w')

import pickle

# Dictionary LOAD
with open('data/pickle/category_to_idx_v3.pickle', 'rb') as fr:
    category_index = pickle.load(fr)


keys = category_index.keys()

for key in keys:
    print(key, category_index[key])