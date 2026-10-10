#################################################################################
#        Análise e Classificação de Sentimentos em Manchetes Jornalísticas      #
# Autores:                                                                      #
# Suiany Pinto Gomes - 582147                                                   #
# Dayane Magalhães Ferreira - 587972                                            #
# Hellen Lavyne Sousa Vasconselos - 580524                                      #
#################################################################################

import pandas as pd
import re

df =  pd.read_csv('brazilian_headlines_sentiments.csv')
df = df.copy()
print(df.head())

# Remover coluna de índice original do CSV
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("Dimensões após remoção:", df.shape)
print(df.columns.tolist())
print(df.head())

# Traduzindo colunas apenas para sabermos com quais variáveis estamos lidando (o df não foi modificado)
colunas_portugues = ['Site', 'Palavras-Chave', 'Manchete Português', 'Manchete Inglês',
                     'Pontuação de Sentimento port', 'Pontuação de Sentimento ing', 'Magnitude Sentimento port',
                     'Magnitude Sentimento ing', 'Estava Online', 'Data de Publicação', 'Data de Remoção', 'Tempo Total Disponível (ms)']

print(f"Colunas Originais:\n{df.columns.value_counts()}")
print("\nColunas Traduzidas:\n")
print(colunas_portugues)

# Verificando informações do dataset
df.info()

df.isnull().sum()

print("Nulos em onlineEndDate:")
print(df['onlineEndDate'].isnull().sum())

print("\nNulos em onlineTotalTimeMS:")
print(df['onlineTotalTimeMS'].isnull().sum())


print(pd.crosstab(
    df['isOnline'],
    df['onlineEndDate'].isnull()
))

print(pd.crosstab(
    df['isOnline'],
    df['onlineTotalTimeMS'].isnull()
))

print(
    "Nulos nas duas colunas:",
    (df['onlineEndDate'].isnull() & df['onlineTotalTimeMS'].isnull()).sum()
)

# Verificar manchetes duplicadas
duplicadas = df[df['headlinePortuguese'].duplicated(keep=False)].copy()

print("Total de registros envolvidos em duplicatas:", len(duplicadas))
print("Número de manchetes duplicadas:", duplicadas['headlinePortuguese'].nunique())

# Verificar se uma mesma manchete possui diferentes valores de sentimento
duplicatas_classes = (
    duplicadas.groupby('headlinePortuguese')['sentimentScorePortuguese'].nunique())

print("Manchetes duplicadas com mais de um valor de sentimento:",
    (duplicatas_classes > 1).sum())

print("Manchetes duplicadas com um único valor de sentimento:",
    (duplicatas_classes == 1).sum())

# Manchetes duplicadas que possuem mais de um valor de sentimento
manchetes_conflitantes = duplicatas_classes[duplicatas_classes > 1].index

conflitos = (
    duplicadas[
        duplicadas['headlinePortuguese'].isin(manchetes_conflitantes)
    ][['headlinePortuguese', 'sentimentScorePortuguese']]
    .sort_values('headlinePortuguese')
)

print("Total de registros envolvidos nos conflitos:",
      len(conflitos))

print("Número de manchetes conflitantes:",
      conflitos['headlinePortuguese'].nunique())

# Criar a classe de sentimento
duplicadas['sentimentClass'] = duplicadas['sentimentScorePortuguese'].apply(
    lambda x: 'negative' if x < 0
    else ('positive' if x > 0 else 'neutral')
)

# Verificar as classes associadas às manchetes conflitantes
conflitos_classes = (
    duplicadas[duplicadas['headlinePortuguese'].isin(manchetes_conflitantes)
    ].groupby('headlinePortuguese')['sentimentClass'].nunique())

print(
    "Manchetes duplicadas com mais de uma classe:",
    (conflitos_classes > 1).sum())

# Mostrar as manchetes que realmente possuem classes diferentes
conflitos_de_classe = conflitos_classes[conflitos_classes > 1]

print(conflitos_de_classe)

print(df.describe())
print("\nEstatísticas:")

######### PRÉ-PROCESSAMENTO ##########
# Antes do pré-processamento textual
print(df['headlinePortuguese'])

# Antes da normalização
print("ANTES:")
print(df['headlinePortuguese'].head())

# Normalização: transformar para letras minúsculas
df['headlinePortuguese'] = df['headlinePortuguese'].str.lower()

# Depois da normalização
print("\nDEPOIS:")
print(df['headlinePortuguese'].head())

# Verificar se ainda existem letras maiúsculas
tem_maiuscula = df['headlinePortuguese'].str.contains(
    r'[A-Z]', regex=True, na=False)

print("Quantidade de manchetes com letras maiúsculas:", tem_maiuscula.sum())


# Verificar se existem espaços no início ou no final
espacos_extremos = df['headlinePortuguese'].str.match(
    r'^\s|\s$', na=False)

print(
    "Manchetes com espaços no início ou no final:",
    espacos_extremos.sum())

# Verificar se existem espaços repetidos
espacos_repetidos = df['headlinePortuguese'].str.contains(
    r'\s{2,}', regex=True, na=False)

print("Manchetes com espaços repetidos:", espacos_repetidos.sum())

# Remover espaços no início e no final
df['headlinePortuguese'] = df['headlinePortuguese'].str.strip()

# Substituir espaços consecutivos por um único espaço
df['headlinePortuguese'] = df['headlinePortuguese'].str.replace(
    r'\s+', ' ', regex=True)

print(df['headlinePortuguese'].head())

# Verificar se ainda existem espaços no início ou no final
espacos_extremos = df['headlinePortuguese'].str.match(
    r'^\s|\s$', na=False)

# Verificar se ainda existem espaços repetidos
espacos_repetidos = df['headlinePortuguese'].str.contains(
    r'\s{2,}', regex=True, na=False)

print(
    "Manchetes com espaços no início ou no final:",
    espacos_extremos.sum())

print(
    "Manchetes com espaços repetidos:",
    espacos_repetidos.sum())

# Identificar caracteres especiais presentes nas manchetes
caracteres_especiais = df['headlinePortuguese'].str.findall(
    r'[^a-zA-ZÀ-ÿ0-9\s]'
)

# Juntar todos os caracteres encontrados
todos_especiais = []

for lista in caracteres_especiais:
    todos_especiais.extend(lista)

# Mostrar os caracteres especiais encontrados
print("Caracteres especiais encontrados:")
print(set(todos_especiais))

import unicodedata

# Normalização Unicode
df['headlinePortuguese'] = df['headlinePortuguese'].apply(
    lambda texto: unicodedata.normalize('NFC', texto)
)

print(df['headlinePortuguese'].head())

# ==========================================
# REMOÇÃO DE RUÍDO
# ==========================================

import string

# Remover pontuação e símbolos das manchetes
df['headlinePortuguese'] = df['headlinePortuguese'].apply(
    lambda texto: ''.join(
        caractere for caractere in texto
        if caractere not in string.punctuation
    )
)

# Verificar o resultado
print("Manchetes após a remoção de ruído:")
print(df['headlinePortuguese'].head())

# Verificar se ainda existem caracteres especiais

caracteres_restantes = df['headlinePortuguese'].str.findall(
    r'[^a-zA-ZÀ-ÿ0-9\s]'
)

todos_restantes = []

for lista in caracteres_restantes:
    todos_restantes.extend(lista)

print("Caracteres especiais restantes:")
print(set(todos_restantes))

# ==========================================
# REMOÇÃO DOS CARACTERES ESPECIAIS RESTANTES
# ==========================================

caracteres_ruido = '“”‘’…—–🎧\u200bºª°'

df['headlinePortuguese'] = df['headlinePortuguese'].apply(
    lambda texto: ''.join(
        caractere for caractere in texto
        if caractere not in caracteres_ruido
    )
)

# Verificar o resultado
print("Manchetes após a remoção dos caracteres restantes:")
print(df['headlinePortuguese'].head())

# ==========================================
# VERIFICAÇÃO DA REMOÇÃO DE RUÍDO
# ==========================================

# Identificar caracteres especiais que ainda existem
caracteres_restantes = df['headlinePortuguese'].str.findall(
    r'[^a-zA-ZÀ-ÿ0-9\s]'
)

# Juntar todos os caracteres encontrados
todos_restantes = []

for lista in caracteres_restantes:
    todos_restantes.extend(lista)

# Mostrar o resultado
print("Caracteres especiais restantes:")
print(set(todos_restantes))

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

texto = df['headlinePortuguese'].iloc[0]

tokens = word_tokenize(texto)

print(tokens)

# ==========================================
# TOKENIZAÇÃO
# ==========================================

# Importar a função de tokenização
from nltk.tokenize import word_tokenize

# Aplicar a tokenização em todas as manchetes
df['tokens'] = df['headlinePortuguese'].apply(word_tokenize)

# Verificar as manchetes e seus respectivos tokens
print(df[['headlinePortuguese', 'tokens']].head())

# ==========================================
# TESTE: STEMMING x LEMATIZAÇÃO
# ==========================================

# ---------- STEMMING ----------
import nltk
from nltk.stem import RSLPStemmer

nltk.download('rslp')

stemmer = RSLPStemmer()

# Palavras retiradas das primeiras manchetes
palavras = [
    'desculpa',
    'mandar',
    'investiga',
    'defendeu',
    'dá',
    'faça',
    'fui'
]

print("STEMMING:")
for palavra in palavras:
    print(palavra, "→", stemmer.stem(palavra))

# ==========================================
# LEMATIZAÇÃO
# ==========================================

import spacy

# Carregar o modelo de português
nlp = spacy.load("pt_core_news_sm")

print("LEMATIZAÇÃO:")

doc = nlp("desculpa mandar investiga defendeu dá faça fui")

for token in doc:
    print(token.text, "→", token.lemma_)


df['lemmas'] = df['headlinePortuguese'].apply(
    lambda texto: ' '.join(
        token.lemma_ for token in nlp(texto)
    )
)

# Verificar o resultado
print(df[['headlinePortuguese', 'lemmas']].head())

df[['headlinePortuguese', 'lemmas']].head()

# ==========================================
# VERIFICAÇÃO DA LEMATIZAÇÃO
# ==========================================

print("Valores nulos em lemmas:", df['lemmas'].isnull().sum())
print("Quantidade de manchetes:", len(df))
print("Quantidade de lematizações:", len(df['lemmas']))

# Depois do pré-processamento textual
print(df['headlinePortuguese'])

# ==========================================
# VERIFICAÇÃO DO DESBALANCEAMENTO
# ==========================================

print("Distribuição do sentimentScorePortuguese:")
print(df['sentimentScorePortuguese'].describe())

print("\nValores mais frequentes:")
print(df['sentimentScorePortuguese'].value_counts().head(20))

# ==========================================
# VERIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# ==========================================

print("Valores únicos por coluna:\n")

for coluna in df.columns:
  convertida = df[coluna].astype(str)
  print(f"{coluna}: {convertida.nunique()} valores únicos")

def classificar_sentimento(score):
    if score < 0:
        return 'negativo'
    elif score > 0:
        return 'positivo'
    else:
        return 'neutro'

df['sentimento'] = df['sentimentScorePortuguese'].apply(classificar_sentimento)

def classificar_sentimento(score):
    if score < 0:
        return 'negativo'
    elif score > 0:
        return 'positivo'
    else:
        return 'neutro'

df['sentimento'] = df['sentimentScorePortuguese'].apply(classificar_sentimento)

print("Distribuição das classes:")
print(df['sentimento'].value_counts())

print("\nDistribuição percentual:")
print((df['sentimento'].value_counts(normalize=True) * 100).map('{:.2f}%'.format))

# Verificando o tamanho do corpus novamente
print(f"O corpus possui {df.shape[0]} registros e {df.shape[1]} colunas.")

# Verificando a quantidade de classes:
print("Número de classes:", df['sentimento'].nunique())

# Verificando quais foram as classes criadas
print(f"classes: {df['sentimento'].unique()}")

# Verificando a distribuição em cada classe
print(f"\nDistribuição das Classes: \n{df['sentimento'].value_counts()}")
print(f"\nDistribuição Percentual das Classes: \n{(df['sentimento'].value_counts(normalize=True) * 100).map('{:.2f}%'.format)}")

# ==========================================
# DISTRIBUIÇÃO DAS CLASSES
# ==========================================

import matplotlib.pyplot as plt

distribuicao_classes = df['sentimento'].value_counts()
percentuais = df['sentimento'].value_counts(normalize=True) * 100

print("Distribuição das classes:")
print(distribuicao_classes)

print("\nDistribuição percentual:")
print(percentuais.round(2))

plt.figure(figsize=(10, 8))

barras = plt.bar(
    distribuicao_classes.index,
    distribuicao_classes.values
)

plt.title('Distribuição das classes de sentimento')
plt.xlabel('Classe')
plt.ylabel('Quantidade de manchetes')

for barra, percentual in zip(barras, percentuais):
    valor = barra.get_height()

    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f'{int(valor)}\n({percentual:.2f}%)',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords_pt = set(stopwords.words('portuguese'))
print("Quantidade de stopwords", len(stopwords_pt))

# ==========================================
# FREQUÊNCIA DAS PALAVRAS
# ==========================================

from collections import Counter
from nltk.corpus import stopwords

# Lista de stopwords em português
stopwords_pt = set(stopwords.words('portuguese'))

frequencia_palavras = Counter()

for texto in df['lemmas'].fillna(''):
    for palavra in texto.split():

        palavra = palavra.lower()

        if palavra not in stopwords_pt:
            frequencia_palavras[palavra] += 1

print("50 palavras mais frequentes:")

for palavra, frequencia in frequencia_palavras.most_common(50):
    print(f"{palavra}: {frequencia}")

# ==========================================
# VISUALIZAÇÃO DAS 50 PALAVRAS MAIS FREQUENTES
# ==========================================

top_50 = frequencia_palavras.most_common(50)

palavras = [item[0] for item in top_50]
frequencias = [item[1] for item in top_50]

plt.figure(figsize=(10, 7))

barras = plt.barh(
    palavras[::-1],
    frequencias[::-1]
)

plt.title('50 palavras mais frequentes no corpus')
plt.xlabel('Quantidade de ocorrências')
plt.ylabel('Palavra')

# Exibe a quantidade no final de cada barra
for barra in barras:
    valor = barra.get_width()

    plt.text(
        valor + 2,
        barra.get_y() + barra.get_height() / 2,
        f'{int(valor)}',
        va='center'
    )

plt.tight_layout()
plt.show()

# ==========================================
# VISUALIZAÇÃO DAS 50 PALAVRAS MAIS FREQUENTES
# ==========================================

top_20 = frequencia_palavras.most_common(20)

palavras = [item[0] for item in top_20]
frequencias = [item[1] for item in top_20]

plt.figure(figsize=(10, 7))

barras = plt.barh(
    palavras[::-1],
    frequencias[::-1]
)

plt.title('20 palavras mais frequentes no corpus')
plt.xlabel('Quantidade de ocorrências')
plt.ylabel('Palavra')

# Exibe a quantidade no final de cada barra
for barra in barras:
    valor = barra.get_width()

    plt.text(
        valor + 2,
        barra.get_y() + barra.get_height() / 2,
        f'{int(valor)}',
        va='center'
    )

plt.tight_layout()
plt.show()

# ==========================================
# NUVEM DE PALAVRAS
# ==========================================

from wordcloud import WordCloud

nuvem = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate_from_frequencies(frequencia_palavras)

plt.figure(figsize=(12, 6))

plt.imshow(nuvem, interpolation='bilinear')

plt.axis('off')
plt.title('Nuvem de palavras do corpus')

plt.show()

# ==========================================
# CONVERSÃO DO CORPUS PARA DADOS NUMÉRICOS
# TF-IDF
# ==========================================

# Para a vetorização, vamos utilizar a representação textual pré-processada, e não a manchete original.
# Como lemmas está armazenada como texto, podemos utilizá-la diretamente.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words=list(stopwords_pt),
    token_pattern=r'(?u)\b[^\W\d_]+\b'
)

X_tfidf = tfidf.fit_transform(
    df['lemmas'].fillna('')
)

print("Dimensões da matriz TF-IDF:", X_tfidf.shape)
print("Quantidade de termos:", len(tfidf.get_feature_names_out()))

# Exibindo 5 documentos e 10 termos
print("\nExibindo 5 documentos e 10 termos 5×10:")
X_tfidf[:5, :10].toarray()

# Verificando se a matriz está sendo corretamente interpretada
print("Número de valores diferentes de zero:", X_tfidf.nnz)

total_elementos = X_tfidf.shape[0] * X_tfidf.shape[1]

print("Total de elementos:", total_elementos)
print("Elementos diferentes de zero:", X_tfidf.nnz)

# Selecionando os termos que aparecem na primeira manchete

indices = X_tfidf[0].nonzero()[1]

print("\nTermos presentes na primeira manchete:")

for indice in indices:
    termo = tfidf.get_feature_names_out()[indice]
    valor = X_tfidf[0, indice]

    print(f"{termo}: {valor:.4f}")

# ==========================================
# CONVERSÃO DO CORPUS - COUNT VECTORIZER
# ==========================================

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(
    stop_words=list(stopwords_pt),
    token_pattern=r'(?u)\b[^\W\d_]+\b'
)

X_count = count_vectorizer.fit_transform(
    df['lemmas'].fillna('')
)

print("Dimensões da matriz CountVectorizer:", X_count.shape)
print("Quantidade de termos:", len(count_vectorizer.get_feature_names_out()))

# Metodo do cotovelo para definir o k
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inercia = []
valores_k = range(2, 11)

for k_atual in valores_k:
    modelo = KMeans(
        n_clusters=k_atual,
        random_state=42,
        n_init=10
    )

    modelo.fit(X_tfidf)
    inercia.append(modelo.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    valores_k,
    inercia,
    marker='o'
)

plt.title('Método do Cotovelo — K-means')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inércia')
plt.xticks(valores_k)

plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

valores_k_comparacao = [4, 5, 6]

resultados_k = []

for k in valores_k_comparacao:
    kmeans_teste = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_teste = kmeans_teste.fit_predict(X_tfidf)

    inercia = kmeans_teste.inertia_

    silhouette = silhouette_score(
        X_tfidf,
        labels_teste,
        sample_size=2000,
        random_state=42
    )

    resultados_k.append({
        'k': k,
        'inercia': inercia,
        'silhouette': silhouette
    })

tabela_k = pd.DataFrame(resultados_k)

print(tabela_k)

from sklearn.cluster import KMeans

melhor_k = 6

kmeans_final = KMeans(
    n_clusters=melhor_k,
    random_state=42,
    n_init=10
)

df['cluster_kmeans'] = kmeans_final.fit_predict(X_tfidf)

print("K-means finalizado com k =", melhor_k)

contagem_clusters = df['cluster_kmeans'].value_counts().sort_index()

print("Quantidade de documentos por cluster:")
print(contagem_clusters)

percentual_clusters = (
    df['cluster_kmeans']
    .value_counts(normalize=True)
    .sort_index() * 100
)

tabela_clusters = pd.DataFrame({
    'Quantidade': contagem_clusters,
    'Percentual (%)': percentual_clusters.round(2)
})

print(tabela_clusters)

termos = tfidf.get_feature_names_out()
centroides = kmeans_final.cluster_centers_

for i in range(melhor_k):
    indices = centroides[i].argsort()[::-1][:15]

    print(f"\nCluster {i}:")
    print(", ".join(termos[indices]))

contagem_clusters = (
    df['cluster_kmeans']
    .value_counts()
    .sort_index()
)

percentual_clusters = (
    df['cluster_kmeans']
    .value_counts(normalize=True)
    .sort_index() * 100
)

tabela_clusters = pd.DataFrame({
    'Quantidade': contagem_clusters,
    'Percentual (%)': percentual_clusters.round(2)
})

print(tabela_clusters)

tabela_cluster_sentimento = pd.crosstab(
    df['cluster_kmeans'],
    df['sentimento']
)

print(tabela_cluster_sentimento, "\n")

tabela_cluster_sentimento_pct = pd.crosstab(
    df['cluster_kmeans'],
    df['sentimento'],
    normalize='index'
) * 100

print("\n","Tabela Percentual\n", tabela_cluster_sentimento_pct.round(2), "\n")

tabela_cluster_sentimento_pct = tabela_cluster_sentimento_pct[
    ['negativo', 'neutro', 'positivo']
]

print("\n", tabela_cluster_sentimento_pct.round(2))

ax = tabela_cluster_sentimento_pct.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6)
)

plt.title('Distribuição do sentimento dentro de cada cluster')
plt.xlabel('Cluster')
plt.ylabel('Percentual (%)')
plt.xticks(rotation=0)
plt.legend(title='Sentimento')

plt.tight_layout()
plt.show()

silhouette_scores = []

valores_k = range(2, 21)

for k in valores_k:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_tfidf)

    score = silhouette_score(
        X_tfidf,
        labels,
        metric='euclidean',
        sample_size=2000,
        random_state=42
    )

    silhouette_scores.append(score)

    print(f"k={k}: Silhouette Score = {score:.4f}")

plt.figure(figsize=(9, 5))

plt.plot(
    valores_k,
    silhouette_scores,
    marker='o'
)

plt.title('Silhouette Score para diferentes valores de k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(valores_k)

plt.tight_layout()
plt.show()

import numpy as np
melhor_indice = np.argmax(silhouette_scores)

print(
    "Melhor k:",
    list(valores_k)[melhor_indice]
)

print(
    "Maior Silhouette Score:",
    round(silhouette_scores[melhor_indice], 4)
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Amostra aleatória fixa
rng = np.random.RandomState(42)

indices_amostra = rng.choice(
    X_tfidf.shape[0],
    size=150,
    replace=False
)

amostra_tfidf = X_tfidf[indices_amostra].toarray()

# Testar os três métodos
metodos = ['single', 'complete', 'average']

for metodo in metodos:

    Z = linkage(
        amostra_tfidf,
        method=metodo,
        metric='euclidean'
    )

    plt.figure(figsize=(14, 6))

    dendrogram(
        Z,
        no_labels=True,
        truncate_mode='lastp',
        p=30
    )

    plt.title(f'Dendrograma - Método de Ligação: {metodo}')
    plt.xlabel('Grupos/Amostras')
    plt.ylabel('Distância Euclidiana')

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

# Mesma amostra aleatória fixa
rng = np.random.RandomState(42)

indices_amostra = rng.choice(
    X_tfidf.shape[0],
    size=150,
    replace=False
)

amostra_tfidf = X_tfidf[indices_amostra].toarray()

# Clusterização hierárquica - Complete
Z_complete = linkage(
    amostra_tfidf,
    method='complete',
    metric='euclidean'
)

# Tentativa de obter até 6 clusters
clusters_hierarquicos = fcluster(
    Z_complete,
    t=6,
    criterion='maxclust'
)

print("Quantidade de clusters:", len(np.unique(clusters_hierarquicos)))

print("\nQuantidade de observações em cada cluster:")
print(
    pd.Series(clusters_hierarquicos)
    .value_counts()
    .sort_index()
)

# ==========================================
# APLICAÇÃO 3: IDENTIFICAÇÃO DE MANCHETES SEMELHANTES
# ==========================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Seleciona a manchete que será consultada
indice_consulta = 0

manchete_consulta = df.iloc[indice_consulta]['headlinePortuguese']
lema_consulta = str(df.iloc[indice_consulta]['lemmas']).lower().strip()

# 2. Calcula a similaridade entre a consulta e todas as manchetes
# ESTA ETAPA CRIA A VARIÁVEL "similaridades"
similaridades = cosine_similarity(
    X_tfidf[indice_consulta],
    X_tfidf
).flatten()

# 3. Normaliza os textos para identificar duplicatas
manchetes_normalizadas = (
    df['headlinePortuguese']
    .fillna('')
    .astype(str)
    .str.lower()
    .str.strip()
    .to_numpy()
)

lemas_normalizados = (
    df['lemmas']
    .fillna('')
    .astype(str)
    .str.lower()
    .str.strip()
    .to_numpy()
)

# 4. Ordena os índices da maior para a menor similaridade
ordem_similaridade = np.argsort(similaridades)[::-1]

# 5. Seleciona cinco manchetes distintas
indices_semelhantes = []
manchetes_vistas = set()
lemas_vistos = set()

for i in ordem_similaridade:

    # Não inclui a própria manchete consultada
    if i == indice_consulta:
        continue

    # Ignora similaridades iguais a zero
    if similaridades[i] <= 0:
        continue

    # Evita incluir novamente a manchete consultada
    if manchetes_normalizadas[i] == manchetes_normalizadas[indice_consulta]:
        continue

    # Evita duplicatas textuais exatas
    if manchetes_normalizadas[i] in manchetes_vistas:
        continue

    # Evita textos com a mesma representação lematizada
    if lemas_normalizados[i] == lema_consulta:
        continue

    if lemas_normalizados[i] in lemas_vistos:
        continue

    manchetes_vistas.add(manchetes_normalizadas[i])
    lemas_vistos.add(lemas_normalizados[i])

    indices_semelhantes.append(i)

    if len(indices_semelhantes) == 5:
        break

# 6. Cria a tabela com os resultados
resultado_similares = pd.DataFrame({
    'Manchete': [
        df.iloc[i]['headlinePortuguese']
        for i in indices_semelhantes
    ],
    'Similaridade (%)': [
        round(float(similaridades[i]) * 100, 2)
        for i in indices_semelhantes
    ]
})
from IPython.display import display

# 7. Exibe os resultados
pd.set_option('display.max_colwidth', None)

print("MANCHETE CONSULTADA:")
print(manchete_consulta)

print("\nCINCO MANCHETES DISTINTAS MAIS SEMELHANTES:")

display(resultado_similares)

from sklearn.model_selection import GroupShuffleSplit

X_texto = df['lemmas'].fillna('')
y = df['sentimento']

# Cada manchete lematizada representa um grupo
groups = X_texto

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X_texto, y, groups=groups)
)

X_train_texto = X_texto.iloc[train_idx]
X_test_texto = X_texto.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Treino:", len(X_train_texto))
print("Teste:", len(X_test_texto))

print("Distribuição no conjunto de treino:")
print(y_train.value_counts())
print("\nPercentual no conjunto de treino:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nDistribuição no conjunto de teste:")
print(y_test.value_counts())
print("\nPercentual no conjunto de teste:")
print((y_test.value_counts(normalize=True) * 100).round(2))

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_classificacao = TfidfVectorizer(
    stop_words=list(stopwords_pt),
    token_pattern=r'(?u)\b[^\W\d_]+\b'
)

X_train_tfidf = tfidf_classificacao.fit_transform(X_train_texto)
X_test_tfidf = tfidf_classificacao.transform(X_test_texto)

print("TF-IDF - treino:", X_train_tfidf.shape)
print("TF-IDF - teste:", X_test_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB

modelo_nb = MultinomialNB()

modelo_nb.fit(
    X_train_tfidf,
    y_train
)

y_pred_nb = modelo_nb.predict(X_test_tfidf)

print("Modelo Naive Bayes treinado com sucesso.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Precision
precision_nb = precision_score(
    y_test,
    y_pred_nb,
    average='weighted'
)

# Recall
recall_nb = recall_score(
    y_test,
    y_pred_nb,
    average='weighted'
)

# F1-score
f1_nb = f1_score(
    y_test,
    y_pred_nb,
    average='weighted'
)

print("Avaliação — Naive Bayes")
print(f"Accuracy : {accuracy_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall   : {recall_nb:.4f}")
print(f"F1-score : {f1_nb:.4f}")

print("\nRelatório de classificação:")
print(
    classification_report(
        y_test,
        y_pred_nb,
        digits=4
    )
)

#-----------------------------------------
#   Matriz de confusão
#-----------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm_nb = confusion_matrix(
    y_test,
    y_pred_nb,
    labels=['negativo', 'neutro', 'positivo']
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb,
    display_labels=['Negativo', 'Neutro', 'Positivo']
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format='d'
)

plt.title('Matriz de Confusão — Naive Bayes')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression

modelo_lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

modelo_lr.fit(
    X_train_tfidf,
    y_train
)

y_pred_lr = modelo_lr.predict(X_test_tfidf)

print("Modelo de Regressão Logística treinado com sucesso.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Precision
precision_lr = precision_score(
    y_test,
    y_pred_lr,
    average='weighted'
)

# Recall
recall_lr = recall_score(
    y_test,
    y_pred_lr,
    average='weighted'
)

# F1-score
f1_lr = f1_score(
    y_test,
    y_pred_lr,
    average='weighted'
)

print("Avaliação — Regressão Logística")
print(f"Accuracy : {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall   : {recall_lr:.4f}")
print(f"F1-score : {f1_lr:.4f}")

print("\nRelatório de classificação:")
print(
    classification_report(
        y_test,
        y_pred_lr,
        digits=4
    )
)

#-----------------------------------------
#   Matriz de confusão
#-----------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr,
    labels=['negativo', 'neutro', 'positivo']
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=['Negativo', 'Neutro', 'Positivo']
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format='d'
)

plt.title('Matriz de Confusão — Regressão Logística + TF-IDF')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

count_classificacao = CountVectorizer(
    stop_words=list(stopwords_pt),
    token_pattern=r'(?u)\b[^\W\d_]+\b'
)

X_train_count = count_classificacao.fit_transform(X_train_texto)

X_test_count = count_classificacao.transform(X_test_texto)

print("CountVectorizer - treino:", X_train_count.shape)
print("CountVectorizer - teste:", X_test_count.shape)

from sklearn.naive_bayes import MultinomialNB

modelo_nb_count = MultinomialNB()

modelo_nb_count.fit(
    X_train_count,
    y_train
)

y_pred_nb_count = modelo_nb_count.predict(
    X_test_count
)

print("Modelo Naive Bayes + CountVectorizer treinado com sucesso.")

accuracy_nb_count = accuracy_score(
    y_test,
    y_pred_nb_count
)

precision_nb_count = precision_score(
    y_test,
    y_pred_nb_count,
    average='weighted'
)

recall_nb_count = recall_score(
    y_test,
    y_pred_nb_count,
    average='weighted'
)

f1_nb_count = f1_score(
    y_test,
    y_pred_nb_count,
    average='weighted'
)

print("Avaliação — Naive Bayes + CountVectorizer")
print(f"Accuracy : {accuracy_nb_count:.4f}")
print(f"Precision: {precision_nb_count:.4f}")
print(f"Recall   : {recall_nb_count:.4f}")
print(f"F1-score : {f1_nb_count:.4f}")

print("\nRelatório de classificação:")
print(
    classification_report(
        y_test,
        y_pred_nb_count,
        digits=4
    )
)

#---------------------------------------
#   MATRIZ DE CONFUSÃO
#---------------------------------------

cm_nb_count = confusion_matrix(
    y_test,
    y_pred_nb_count,
    labels=['negativo', 'neutro', 'positivo']
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb_count,
    display_labels=['Negativo', 'Neutro', 'Positivo']
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format='d'
)

plt.title('Matriz de Confusão — Naive Bayes + CountVectorizer')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression

modelo_lr_count = LogisticRegression(
    max_iter=1000,
    random_state=42
)

modelo_lr_count.fit(
    X_train_count,
    y_train
)

y_pred_lr_count = modelo_lr_count.predict(
    X_test_count
)

print("Modelo de Regressão Logística + CountVectorizer treinado com sucesso.")

accuracy_lr_count = accuracy_score(
    y_test,
    y_pred_lr_count
)

precision_lr_count = precision_score(
    y_test,
    y_pred_lr_count,
    average='weighted'
)

recall_lr_count = recall_score(
    y_test,
    y_pred_lr_count,
    average='weighted'
)

f1_lr_count = f1_score(
    y_test,
    y_pred_lr_count,
    average='weighted'
)

print("Avaliação — Regressão Logística + CountVectorizer")
print(f"Accuracy : {accuracy_lr_count:.4f}")
print(f"Precision: {precision_lr_count:.4f}")
print(f"Recall   : {recall_lr_count:.4f}")
print(f"F1-score : {f1_lr_count:.4f}")

print("\nRelatório de classificação:")
print(
    classification_report(
        y_test,
        y_pred_lr_count,
        digits=4
    )
)

#---------------------------------------
#   MATRIZ DE CONFUSÃO
#---------------------------------------

cm_lr_count = confusion_matrix(
    y_test,
    y_pred_lr_count,
    labels=['negativo', 'neutro', 'positivo']
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr_count,
    display_labels=['Negativo', 'Neutro', 'Positivo']
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format='d'
)

plt.title('Matriz de Confusão — Regressão Logística + CountVectorizer')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.show()

import pandas as pd

tabela_modelos = pd.DataFrame({
    'Modelo': [
        'Naive Bayes',
        'Naive Bayes',
        'Regressão Logística',
        'Regressão Logística'
    ],

    'Representação': [
        'TF-IDF',
        'CountVectorizer',
        'TF-IDF',
        'CountVectorizer'
    ],

    'Accuracy': [
        accuracy_nb,
        accuracy_nb_count,
        accuracy_lr,
        accuracy_lr_count
    ],

    'Precision': [
        precision_nb,
        precision_nb_count,
        precision_lr,
        precision_lr_count
    ],

    'Recall': [
        recall_nb,
        recall_nb_count,
        recall_lr,
        recall_lr_count
    ],

    'F1-score': [
        f1_nb,
        f1_nb_count,
        f1_lr,
        f1_lr_count
    ]
})

tabela_modelos[
    ['Accuracy', 'Precision', 'Recall', 'F1-score']
] = tabela_modelos[
    ['Accuracy', 'Precision', 'Recall', 'F1-score']
].round(4)

print(tabela_modelos)

metricas = ['Accuracy', 'Precision', 'Recall', 'F1-score']

tabela_grafico = tabela_modelos.set_index(
    ['Modelo', 'Representação']
)[metricas]

ax = tabela_grafico.plot(
    kind='bar',
    figsize=(11, 6)
)

plt.title('Comparação de desempenho dos modelos')
plt.xlabel('Modelo e representação')
plt.ylabel('Valor da métrica')
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.legend(title='Métrica')
plt.tight_layout()
plt.show()

melhor_modelo = tabela_modelos.loc[
    tabela_modelos['F1-score'].idxmax()
]

print("Melhor configuração:")
print(melhor_modelo)