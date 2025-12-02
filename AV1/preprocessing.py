#import nltk
#nltk.download('punkt_tab')
#nltk.download('stopwords')
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from imblearn.under_sampling import RandomUnderSampler
import spacy

# Carregando o dataset
df = pd.read_csv("steam_reviews.csv")
print(df.shape)
print(df.info())

# Tratando valores ausentes
df = df.dropna()

# Convertendo a coluna 'review' para minúsculas
df["review"] = [word.lower() for word in df["review"]]

# Removendo caracteres fora do padrão (mantém letras e espaços)
df["review"] = df["review"].str.replace(r"[^\w\s\"]", " ", regex=True)
df["review"] = df["review"].str.replace(r"\d+", " ", regex=True)  # remove números
df["review"] = df["review"].str.replace(r"\s+", " ", regex=True).str.strip() # Normaliza espaços em branco

# Removendo reviews vazias
df = df[df["review"] != ""]

# Tokenização
df["tokens"] = df["review"].apply(word_tokenize)

# Removendo stopwords
pt_stop_words = stopwords.words('portuguese')
df["no_stop_words"] = df["tokens"].apply(lambda tokens: [word for word in tokens if word not in pt_stop_words])

# Lematização
nlp = spacy.load("pt_core_news_sm")

def lemmatize_spacy(tokens):
    text = " ".join(tokens)
    doc = nlp(text)
    return [t.lemma_ for t in doc if not t.is_space]

df["lemma"] = df["no_stop_words"].apply(lemmatize_spacy)

# Balanceamento das classes
df["voted_up"] = df["voted_up"].astype(int)
target = df["voted_up"].value_counts() # distribuição do target

# Gráfico de barras
ax = target.plot(kind='bar', color=['steelblue', 'salmon'])
for i, value in enumerate(target):
    plt.text(i, value / 2, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Distribuição de Classes (voted_up)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Undersampling
X = df.drop(columns=["voted_up"])
y = df["voted_up"]

undersampler = RandomUnderSampler(random_state=42)

X_res, y_res = undersampler.fit_resample(X, y)

new_target = y_res.value_counts()

# Gráfico de barras
ax = new_target.plot(kind='bar', color=['steelblue', 'salmon'])
for i, value in enumerate(new_target):
    plt.text(i, value / 2, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Distribuição de Classes Balanceadas (voted_up)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Dados balanceados
df_bal = X_res.copy()
df_bal["voted_up"] = y_res

# Movendo colunas
end_cols = ["review", "tokens", "no_stop_words", "lemma", "voted_up"]
initials_cols = [column for column in df_bal.columns if column not in end_cols]
df_bal = df_bal[initials_cols + end_cols]

# Salvando o dataset ajustado
df_bal.to_csv("adjusted_reviews.csv", index=False)