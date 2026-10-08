# Dataset fora do repositório (muito grande): baixe de https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
# e extraia train.ft.txt e test.ft.txt para assets/.

import os
from functools import partial
from itertools import islice
from multiprocessing import Pool

import nltk
from nltk import pos_tag_sents
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

CHUNK_SIZE = 7562
N_WORKERS = max(1, (os.cpu_count() or 1) - 2)
BATCH_CHUNKS = 4 * N_WORKERS
DATASETS = {
    "train": "./assets/train.ft.txt",
    "test": "./assets/test.ft.txt",
}
# 1ª letra da etiqueta Penn Treebank -> classe do WordNet; o resto vira substantivo ("n")
WORDNET_POS = {"J": "a", "V": "v", "R": "r"}


def ensure_nltk_data():
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def count_lines(path):
    with open(path, "rb") as f:
        return sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))


def read_chunks(path, chunk_size=CHUNK_SIZE):
    with open(path, "r", encoding="utf-8") as f:
        while chunk := list(islice(f, chunk_size)):
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
        text = ''.join(char for char in text if char.isalpha() or char.isspace())
        normalized_documents.append(text)
    return normalized_documents


def tokenize_documents(documents):
    tokenized_documents = []
    for doc in documents:
        tokens = word_tokenize(doc)
        tokenized_documents.append(tokens)
    return tokenized_documents


def remove_stopwords(tagged_documents, stop_words):
    filtered_documents = []
    for doc in tagged_documents:
        filtered_tokens = [(token, tag) for token, tag in doc if token not in stop_words]
        filtered_documents.append(filtered_tokens)
    return filtered_documents


def lemmatize_documents(filtered_documents, lemmatizer):
    lemmatized_documents = []
    for tokens in filtered_documents:
        lemmatized_tokens = [lemmatizer.lemmatize(token, WORDNET_POS.get(tag[0], "n"))
                             for token, tag in tokens]
        lemmatized_documents.append(' '.join(lemmatized_tokens))
    return lemmatized_documents


def preprocess_chunk(lines, stop_words, lemmatizer):
    labels, texts = split_labels(lines)
    normalized_documents = normalize_text(texts)
    tokenized_documents = tokenize_documents(normalized_documents)
    # pos_tag antes de remover stopwords: o tagger precisa do contexto da frase inteira
    tagged_documents = pos_tag_sents(tokenized_documents)
    filtered_documents = remove_stopwords(tagged_documents, stop_words)
    lemmatized_documents = lemmatize_documents(filtered_documents, lemmatizer)
    return [f"{label} {doc}\n" for label, doc in zip(labels, lemmatized_documents)]


def preprocessing_pipeline():
    ensure_nltk_data()
    process = partial(preprocess_chunk, stop_words=set(stopwords.words('english')),
                      lemmatizer=WordNetLemmatizer())

    output_paths = {}
    with Pool(N_WORKERS) as pool:
        for name, input_path in DATASETS.items():
            output_path = input_path.replace(".ft.txt", ".processed.txt")
            with open(output_path, "w", encoding="utf-8") as output_file, \
                    tqdm(total=count_lines(input_path), desc=name, unit="docs") as progress:
                chunks = read_chunks(input_path)
                # Lotes limitam a RAM (imap sozinho leria o arquivo inteiro). imap mantém a ordem
                # das linhas, que precisa casar com o .ft.txt (análise de erros em 3_model_selection.py)
                while batch := list(islice(chunks, BATCH_CHUNKS)):
                    for processed in pool.imap(process, batch):
                        output_file.writelines(processed)
                        progress.update(len(processed))
            output_paths[name] = output_path

    return output_paths


if __name__ == "__main__":
    processed_data = preprocessing_pipeline()
