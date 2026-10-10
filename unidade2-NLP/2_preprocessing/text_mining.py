from src.utils import get_sample_article

"""
Atividade de mineração de texto
- Tokenização por sentença (cada sentença = um documento = lista de tokens)
- Pré-processamento: minúsculas, remoção de acentos opcional, pontuação,
  números e stopwords
- Funções: TF, DF, IDF, TF-IDF
- Saída: DataFrame (linhas = sentenças, colunas = termos) salvo em CSV
"""
import math
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# TEXTO A SER MINERADO (troque aqui ou passe um arquivo .txt como argumento)
# ---------------------------------------------------------------------------
TEXTO = """
A mineração de texto transforma texto não estruturado em informação útil.
Para analisar documentos, o texto precisa passar por um pré-processamento.
O pré-processamento remove pontuação, números e palavras muito comuns.
Depois da limpeza, cada sentença é transformada em uma lista de tokens.
A frequência dos termos mostra quais palavras aparecem mais em cada documento.
O TF-IDF pondera a frequência do termo pela raridade do termo na coleção.
Termos raros em poucos documentos recebem pesos maiores na matriz TF-IDF.
A matriz final pode ser usada em classificação, agrupamento e busca de texto.
"""

STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem
sob sobre entre e ou mas que se como ao aos à às pelo pela pelos pelas é são
foi ser ter há muito muitos muita muitas mais menos já também só até depois
cada pode ainda isso esse essa esses essas este esta estes estas seu sua seus
suas ele ela eles elas
""".split())


# ---------------------------------------------------------------------------
# Tokenização e pré-processamento
# ---------------------------------------------------------------------------
def tokenizar_sentencas(texto):
    """Divide o texto em sentenças (., !, ?, quebras de linha)."""
    partes = re.split(r"(?<=[.!?])\s+|\n+", texto.strip())
    return [p.strip() for p in partes if p.strip()]


def preprocessar(sentenca):
    """Sentença -> lista de tokens limpos."""
    s = sentenca.lower()
    s = re.sub(r"[^\w\s-]", " ", s)       # remove pontuação
    s = re.sub(r"\d+", " ", s)            # remove números
    tokens = [t.strip("-_") for t in s.split()]
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calcular_tf(documento):
    """TF(t, d) = ocorrências de t em d / total de tokens de d."""
    contagem = Counter(documento)
    total = len(documento)
    return {t: c / total for t, c in contagem.items()} if total else {}


def calcular_df(documentos):
    """DF(t) = número de documentos que contêm t."""
    df = Counter()
    for doc in documentos:
        df.update(set(doc))
    return dict(df)


def calcular_idf(documentos, df=None):
    """IDF(t) = log((1 + N) / (1 + DF(t))) + 1  (suavizado)."""
    n = len(documentos)
    df = df or calcular_df(documentos)
    return {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}


def calcular_tfidf(documentos):
    """Retorna DataFrame (documentos x termos) com TF-IDF."""
    df = calcular_df(documentos)
    idf = calcular_idf(documentos, df)
    vocab = sorted(idf)
    linhas = []
    for doc in documentos:
        tf = calcular_tf(doc)
        linhas.append([tf.get(t, 0.0) * idf[t] for t in vocab])
    return pd.DataFrame(linhas, columns=vocab), df, idf


# ---------------------------------------------------------------------------
def main():
    texto = TEXTO
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            texto = f.read()

    sentencas = tokenizar_sentencas(texto)
    documentos = [preprocessar(s) for s in sentencas]

    matriz, df, idf = calcular_tfidf(documentos)
    matriz.insert(0, "sentenca", sentencas)

    matriz.to_csv("tfidf_matriz.csv", index=False, encoding="utf-8-sig")

    print(f"{len(sentencas)} sentenças | {matriz.shape[1] - 1} termos")
    print("\nTokens por sentença:")
    for i, d in enumerate(documentos):
        print(f"  [{i}] {d}")
    print("\nDF (top 10):", sorted(df.items(), key=lambda x: -x[1])[:10])
    print("\nIDF (top 5 menores):", sorted(idf.items(), key=lambda x: x[1])[:5])
    print("\nMatriz TF-IDF salva em tfidf_matriz.csv")


if __name__ == "__main__":
    main()