import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ast

# Carregamento e preparo
df = pd.read_csv("adjusted_reviews.csv")
print("Shape do dataset:", df.shape)

# Convertendo para lista
df["lemma"] = df["lemma"].apply(ast.literal_eval)

# Texto lematizado em formato de string
df["lemma_text"] = df["lemma"].apply(lambda toks: " ".join(toks))

print(df[["lemma_text", "voted_up"]].head())

# Treino e teste
X = df["lemma_text"]
y = df["voted_up"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # mantém a proporção de classes
)

target_names = ["Não recomenda (0)", "Recomenda (1)"]

# TF-IDF e Bag-of-Words

# TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Bag-of-Words (CountVectorizer)
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Modelos com TF-IDF

# Regressão Logística + TF-IDF
lr_tfidf = LogisticRegression()
lr_tfidf.fit(X_train_tfidf, y_train)

predictions_lr_tfidf = lr_tfidf.predict(X_test_tfidf)

print("\nRegressão Logística (TF-IDF)")
print("Acurácia:", accuracy_score(y_test, predictions_lr_tfidf))
print("Relatório de classificação:\n")
print(classification_report(y_test, predictions_lr_tfidf, target_names=target_names))
print("Matriz de confusão:\n", confusion_matrix(y_test, predictions_lr_tfidf))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions_lr_tfidf), annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Regressão Logistica (TF-IDF)')
plt.show()

# Naive bayes + TF-IDF
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

nb_predictions_tfidf = nb_tfidf.predict(X_test_tfidf)

print("\nNaive Bayes (TF-IDF)")
print("Acurácia:", accuracy_score(y_test, nb_predictions_tfidf))
print("Relatório de classificação:\n")
print(classification_report(y_test, nb_predictions_tfidf, target_names=target_names))
print("Matriz de confusão:\n", confusion_matrix(y_test, nb_predictions_tfidf))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, nb_predictions_tfidf), annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Naive Bayes (TF-IDF)')
plt.show()


# Modelos com Bag-of-Words (bow)

# Regressão logística + bow
lr_bow = LogisticRegression(max_iter=1000)
lr_bow.fit(X_train_cv, y_train)

predictions_lr_bow = lr_bow.predict(X_test_cv)

print("\nRegressão Logística (Bag-of-Words)")
print("Acurácia:", accuracy_score(y_test, predictions_lr_bow))
print("Relatório de classificação:\n")
print(classification_report(y_test, predictions_lr_bow, target_names=target_names))
print("Matriz de confusão:\n", confusion_matrix(y_test, predictions_lr_bow))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions_lr_bow), annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Regressão Logistica (BoW)')
plt.show()

# Naive bayes + bow
nb_bow = MultinomialNB()
nb_bow.fit(X_train_cv, y_train)

nb_predictions_bow = nb_bow.predict(X_test_cv)

print("\nNaive Bayes (Bag-of-Words)")
print("Acurácia:", accuracy_score(y_test, nb_predictions_bow))
print("Relatório de classificação:\n")
print(classification_report(y_test, nb_predictions_bow, target_names=target_names))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, nb_predictions_bow), annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Naive Bayes (BoW)')
plt.show()
