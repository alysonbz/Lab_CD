# Dataset muito grande para fazer upload no GitHub, por isso não está presente no repositório.
# Para rodar o código, baixe os arquivos manualmente: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
# Extraia o arquivo zip, e os zips contidos nele para a pasta assets.

# Estrutura:
# assets/
# ├── train.ft.txt
# ├── test.ft.txt
# ├── train.processed.txt  (gerado por este script)
# ├── test.processed.txt   (gerado por este script)

# ~ 30min para rodar

from itertools import islice

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

CHUNK_SIZE = 7562  # Número de linhas a serem lidas por vez
DATASETS = {
    "train": "./assets/train.ft.txt",
    "test": "./assets/test.ft.txt",
}


def ensure_nltk_data():
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def count_lines(path):
    with open(path, "rb") as f:                                   # ~ 1mb por vez
        return sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))


def read_chunks(path, chunk_size=CHUNK_SIZE):
    with open(path, "r", encoding="utf-8") as f:
        while chunk := list(islice(f, chunk_size)):
            # yield para pausar a função
            # chamando de novo ela volta para o while
            yield chunk  


def split_labels(lines):
    labels, texts = [], []
    for line in lines:
        label, text = line.rstrip("\n").split(" ", 1)
        labels.append(label)
        texts.append(text)
    return labels, texts


def normalize_text(raw_documents):
    normalized_documents = []
    for doc in raw_documents:
        text = doc.lower()
        # Remove pontuação, caracteres especiais e numéricos
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        normalized_documents.append(text)
    return normalized_documents


def tokenize_documents(documents):
    tokenized_documents = []
    for doc in documents:
        tokens = word_tokenize(doc)
        tokenized_documents.append(tokens)
    return tokenized_documents


def remove_stopwords(tokenized_documents, stop_words):
    filtered_documents = []
    for doc in tokenized_documents:
        filtered_tokens = [token for token in doc if token not in stop_words]
        filtered_documents.append(filtered_tokens)
    return filtered_documents


def lemmatize_documents(filtered_documents, lemmatizer):
    lemmatized_documents = []
    for tokens in filtered_documents:
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_documents.append(' '.join(lemmatized_tokens))
    return lemmatized_documents


def preprocess_chunk(lines, stop_words, lemmatizer):
    labels, texts = split_labels(lines)
    normalized_documents = normalize_text(texts)
    tokenized_documents = tokenize_documents(normalized_documents)
    filtered_documents = remove_stopwords(tokenized_documents, stop_words)
    lemmatized_documents = lemmatize_documents(filtered_documents, lemmatizer)
    return [f"{label} {doc}\n" for label, doc in zip(labels, lemmatized_documents)]


def preprocessing_pipeline():
    ensure_nltk_data()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    output_paths = {}
    for name, input_path in DATASETS.items():
        output_path = input_path.replace(".ft.txt", ".processed.txt")
        with open(output_path, "w", encoding="utf-8") as output_file, \
                tqdm(total=count_lines(input_path), desc=name, unit="docs") as progress:
            for lines in read_chunks(input_path):
                output_file.writelines(preprocess_chunk(lines, stop_words, lemmatizer))
                progress.update(len(lines))
        output_paths[name] = output_path

    return output_paths


if __name__ == "__main__":
    processed_data = preprocessing_pipeline()
