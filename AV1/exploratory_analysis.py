import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import ast

#Lê o dataset já pré-processado e balanceado
df = pd.read_csv("adjusted_reviews.csv")
print("Tamanho do corpus:", df.shape)
print(df.info())

# Converte para lista
df["lemma"] = df["lemma"].apply(ast.literal_eval)

# Lista achatada
tokens_bow = [
    w
    for toks in df["lemma"]
    for w in toks
    if w.isalpha() and len(w) > 1
]

# Bag of words
bow = Counter(tokens_bow)
top_20 = bow.most_common(20) # seleciona as 20 palavras mais frequentes
print(top_20)

# Separa palavras e frequências em listas para plotar
words = [w for w, f in top_20]
frequencys = [f for w, f in top_20]

# Gráfico de barras do Top 20 do Bag-of-Words
plt.figure(figsize=(10, 5))
plt.bar(range(len(words)), frequencys)
plt.xticks(range(len(words)), words, rotation=45, ha='right')
plt.title("Top 20 palavras mais frequentes (Bag-of-Words)")
plt.xlabel("Palavra")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# Junta todos os tokens filtrados em uma única string
all_words = " ".join(tokens_bow)

# Cria a wordcloud
cloud = WordCloud(width=800, height=400, background_color="black")
cloud = cloud.generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off") # esconde os eixos para focar só na nuvem
plt.tight_layout()
plt.show()