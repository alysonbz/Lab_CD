import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from utils import get_sample_article, get_english_stop_words
from math import log
nltk.download("punkt")
nltk.download("wordnet")


texto = get_sample_article()


english_stop = set(get_english_stop_words())
lemmatizer = WordNetLemmatizer()

def preprocessar_sentenca(sent):
    sent = sent.lower()
    tokens = word_tokenize(sent)

    tokens_alpha = [t for t in tokens if t.isalpha()]
    tokens_sem_stop = [t for t in tokens_alpha if t not in english_stop]
    tokens_lematizados = [lemmatizer.lemmatize(t) for t in tokens_sem_stop]
    print(tokens_lematizados)
    return tokens_lematizados


sentencas = sent_tokenize(texto)

# Cada sentença vira um documento de tokens limpos
documentos = [preprocessar_sentenca(s) for s in sentencas]

print("Documentos (lista de tokens por sentença):")
for i, doc in enumerate(documentos):
    print(f"Doc {i}: {doc}")


def calcular_tf(documento):
    tf = {}
    total = len(documento)
    for token in documento:
        tf[token] = tf.get(token, 0) + 1
    
    for i in tf:
        tf[i] /= total

    return tf 

def calcular_df(documentos):
    df = {}
    
    for doc in documentos:
        tokin_unico = set(doc)
        for token in tokin_unico:
            df[token] = df.get(token, 0) + 1

    return df

def calcular_idf(df, N):
    idf = {}
    for token, freq in df.items():
        idf[token] = log(N/(freq))
    return idf

def calcular_tfidf(tf, idf):
    tfidf = {}
    
    for token in tf:
        tfidf[token] = tf[token] * idf.get(token, 0)

    return tfidf

df_global = calcular_df(documentos)
N = len(documentos)
idf = calcular_idf(df_global, N)

linhas = []
for i, doc in enumerate(documentos):
    tf = calcular_tf(doc)
    tfidf = calcular_tfidf(tf, idf)
    linhas.append(tfidf)

df_final = pd.DataFrame(linhas).fillna(0)
df_final.insert(0, "sentenca_original", sentencas)

print(df_final)

# df_final.to_csv("tfidf_sentencas.csv", index=False)
# print("Arquivo salvo: tfidf_sentencas.csv")
