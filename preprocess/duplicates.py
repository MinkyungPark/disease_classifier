import pandas as pd

df1 = pd.read_csv('true_dataset_bilstm.csv', encoding='utf-8')
df2 = pd.read_csv('true_dataset_cnn1d.csv', encoding='utf-8')

df1 = df1[['sentences','ture_value']]
df2 = df2[['sentences','ture_value']]
print(len(df1))
print(len(df2))
print('-'*100)

df1 = df1.drop_duplicates('sentences', keep='first')
df2 = df2.drop_duplicates('sentences', keep='first')
print(len(df1))
print(len(df2))
print('-'*100)


df = pd.concat([df1,df2], axis=0)

df_dup = df.drop_duplicates('sentences', keep='first')

# df_dup.to_csv('true_dataset_total.csv', encoding='utf-8', index=False)

print(len(df_dup))
