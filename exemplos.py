import re
import pandas as pd

df = pd.read_csv('fakepedia-corpus-v1.csv', sep=';')

print(df.shape)
print(df.head(8517))
print(df.columns)