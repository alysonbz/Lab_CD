import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# ==============================================================================
# CONFIGURAÇÕES INICIAIS E DOWNLOADS NLTK
# ==============================================================================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def limpar_texto(texto):
    """
    Etapa 1: Pré-processamento e Limpeza Textual
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('portuguese'))

    # Normalização e Remoção de Ruído
    texto = re.sub(r'[^a-zA-Záéíóúçãõâêîôû ]', '', str(texto).lower())

    # Tokenização e Remoção de Stopwords
    tokens = texto.split()
    tokens = [word for word in tokens if word not in stop_words]

    # Lematização
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def main():
    print("=" * 60)
    print(" INICIANDO PIPELINE DE NLP - PROJETO ")
    print("=" * 60)

    # ==============================================================================
    # CARREGAMENTO DO DATASET
    # ==============================================================================
    print("\nAbrindo janela para você selecionar o arquivo CSV...")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    caminho_csv = filedialog.askopenfilename(
        title="Selecione o arquivo do dataset (.csv)",
        filetypes=[("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*")]
    )

    if not caminho_csv:
        print("ERRO: Nenhum arquivo foi selecionado. Encerrando o programa.")
        return

    df = pd.read_csv(caminho_csv)
    print(f"Arquivo carregado com sucesso de: {caminho_csv}")

    # ==============================================================================
    # DETECÇÃO AUTOMÁTICA DE COLUNAS (HEURÍSTICA)
    # ==============================================================================
    # 1. Identificando a coluna de Texto
    col_texto = None
    candidatos_texto = ['text', 'texto', 'tweet', 'content', 'mensagem', 'frase']
    for col in df.columns:
        if col.lower() in candidatos_texto:
            col_texto = col
            break
    if not col_texto:
        # Pega a coluna do tipo object/string com o maior número de valores únicos
        cols_obj = df.select_dtypes(include=['object', 'string']).columns
        if len(cols_obj) > 0:
            col_texto = df[cols_obj].nunique().idxmax()
        else:
            col_texto = df.columns[0]  # Fallback extremo

    # 2. Identificando a coluna de Label
    col_label = None
    candidatos_label = ['label', 'class', 'classe', 'target', 'sentiment', 'hateful', 'categoria']
    for col in df.columns:
        if col.lower() in candidatos_label:
            col_label = col
            break
    if not col_label:
        # Pega a coluna com o menor número de valores únicos (característica de classes)
        col_label = df.nunique().idxmin()

    print(f"\n[!] Colunas mapeadas automaticamente:")
    print(f" -> Coluna de Textos: '{col_texto}'")
    print(f" -> Coluna de Classes: '{col_label}'")

    # Renomeia para o padrão do nosso código e remove linhas vazias nessas colunas
    df = df.rename(columns={col_texto: 'texto', col_label: 'label'})
    df = df.dropna(subset=['texto', 'label']).copy()

    # ==============================================================================
    # Etapa 2: ANÁLISE EXPLORATÓRIA DO CORPUS
    # ==============================================================================
    print("\n[+] ETAPA 2: ANÁLISE EXPLORATÓRIA")
    print(f"Tamanho do corpus: {df.shape[0]} instâncias (linhas)")
    print(f"Número de classes: {df['label'].nunique()}")
    print("\nDistribuição de classes (Frequência Absoluta):")
    print(df['label'].value_counts())

    print("\nAplicando pré-processamento textual (aguarde, pode demorar alguns segundos)...")
    df['texto_limpo'] = df['texto'].apply(limpar_texto)

    # Remove textos que ficaram totalmente vazios após a limpeza
    df = df[df['texto_limpo'].str.strip() != '']

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label', palette='viridis')
    plt.title('Distribuição de Classes no Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Frequência')
    plt.show()

    print("Gerando visualizações de palavras...")
    texto_completo = ' '.join(df['texto_limpo'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras Geral do Corpus')
    plt.show()

    todas_palavras = texto_completo.split()
    freq_palavras = Counter(todas_palavras)
    df_freq = pd.DataFrame(freq_palavras.most_common(20), columns=['Palavra', 'Frequência'])

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_freq, x='Frequência', y='Palavra', palette='mako')
    plt.title('Top 20 Palavras Mais Frequentes (Pós-limpeza)')
    plt.xlabel('Frequência')
    plt.ylabel('Palavra')
    plt.show()

    print("\n[INFO PARA RELATÓRIO] - 3 Opções de Aprendizado Não Supervisionado:")
    print("1. Modelagem de Tópicos (LDA): Agrupar documentos por temas latentes.")
    print("2. Clusterização (K-Means): Segmentar os textos em grupos por similaridade.")
    print("3. Detecção de Anomalias (Isolation Forest): Identificar padrões anômalos no corpus.")

    # ==============================================================================
    # Etapas 3 e 4: CONSTRUÇÃO DO MODELO E AVALIAÇÃO QUANTITATIVA
    # ==============================================================================
    print("\n[INFO PARA RELATÓRIO] - Justificativa dos Modelos:")
    print("- Naive Bayes: Baseline rápido e eficiente, ideal para Bag-of-Words.")
    print("- Regressão Logística: Robusto para lidar com os pesos do TF-IDF em vetores esparsos.")

    X = df['texto_limpo']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vetorizadores = {
        'Bag-of-Words (CountVectorizer)': CountVectorizer(),
        'TF-IDF': TfidfVectorizer()
    }

    modelos = {
        'Naive Bayes': MultinomialNB(),
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42)
    }

    resultados_acuracia = {}
    smote = SMOTE(random_state=42)

    print("\n[+] ETAPA 3 e 4: TREINAMENTO, BALANCEAMENTO E AVALIAÇÃO")
    for nome_vet, vet in vetorizadores.items():
        print(f"\n" + "=" * 50)
        print(f"REPRESENTAÇÃO DE FEATURES: {nome_vet}")
        print("=" * 50)

        X_train_vec = vet.fit_transform(X_train)
        X_test_vec = vet.transform(X_test)

        print("[!] Aplicando SMOTE para balanceamento das classes de treino...")
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

        for nome_mod, modelo in modelos.items():
            chave = f"{nome_mod} com {nome_vet}"
            print(f"\n>> Treinando modelo: {nome_mod}...")

            modelo.fit(X_train_res, y_train_res)
            y_pred = modelo.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            resultados_acuracia[chave] = acc

            print(f"\nMétricas Quantitativas (Acurácia, Precision, Recall, F1-score) para {chave}:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=modelo.classes_, yticklabels=modelo.classes_)
            plt.title(f'Matriz de Confusão:\n{chave}')
            plt.ylabel('Classe Verdadeira')
            plt.xlabel('Classe Prevista')
            plt.tight_layout()
            plt.show()

    print("\n" + "=" * 60)
    print(" RESUMO FINAL (ACURÁCIA) PARA A TABELA DO RELATÓRIO ")
    print("=" * 60)
    for k, v in sorted(resultados_acuracia.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()