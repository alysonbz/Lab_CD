"""Análise exploratória do corpus (formato fastText, saída do pré-processamento).

Ao final gera o dataset numérico usado por 2.1_eda_with_models.py e
3_model_selection.py, em assets/:
  {train,test}.bow.npz    contagens (Bag-of-Words), adequado ao Naive Bayes
  {train,test}.tfidf.npz  TF-IDF normalizado, adequado ao SVM
  {train,test}.labels.npy rótulos: 0 = negativo, 1 = positivo
  vocab.txt               termo de cada coluna, um por linha

Dependências: pip install numpy pandas matplotlib scipy scikit-learn wordcloud tqdm
"""
from collections import Counter, defaultdict
from itertools import islice
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
from wordcloud import WordCloud

ASSETS = Path("./assets")
PROCESSED = {
    "train": ASSETS / "train.processed.txt",
    "test": ASSETS / "test.processed.txt",
}
LABEL_NAMES = {"__label__1": "negativo", "__label__2": "positivo"}
LABEL_IDS = {label: i for i, label in enumerate(LABEL_NAMES)}
VOCAB_SIZE = 50_000  # palavras mais frequentes do treino que viram colunas
CHUNK_SIZE = 100_000  # documentos vetorizados por vez
OUT = Path("./outputs")
OUT.mkdir(exist_ok=True)


def name_of(label):
    return LABEL_NAMES.get(label, label)


def count_lines(path):
    with open(path, "rb") as f:
        return sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def scan(path):
    class_words = defaultdict(Counter)
    class_lengths = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=count_lines(path), desc=path.name, unit="docs"):
            label, _, text = line.rstrip("\n").partition(" ")
            tokens = text.split()
            class_words[label].update(tokens)
            class_lengths[label].append(len(tokens))
    return class_words, class_lengths


def summarize(class_words, class_lengths):
    total = Counter()
    for counter in class_words.values():
        total.update(counter)

    all_lens = np.concatenate([np.array(v) for v in class_lengths.values()])
    n_total = len(all_lens)
    rows = []
    for label in sorted(class_words):
        lens = np.array(class_lengths[label])
        rows.append({
            "classe": name_of(label), "documentos": len(lens),
            "% docs": 100 * len(lens) / n_total, "tokens": int(lens.sum()),
            "vocabulário": len(class_words[label]),
            "média tokens/doc": lens.mean(), "mediana": np.median(lens),
            "máx": lens.max(),
        })
    rows.append({
        "classe": "TOTAL", "documentos": n_total, "% docs": 100.0,
        "tokens": int(all_lens.sum()), "vocabulário": len(total),
        "média tokens/doc": all_lens.mean(), "mediana": np.median(all_lens),
        "máx": all_lens.max(),
    })
    hapax = sum(1 for c in total.values() if c == 1)
    return pd.DataFrame(rows), total, hapax


def plot_class_distribution(class_lengths, filename):
    labels = sorted(class_lengths)
    counts = [len(class_lengths[l]) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([name_of(l) for l in labels], counts, color=["#d9534f", "#5cb85c"][:len(labels)])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n({100 * c / sum(counts):.1f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Documentos")
    ax.set_title("Distribuição de classes")
    ax.set_ylim(0, max(counts) * 1.2)
    save(fig, filename)


def plot_length_hist(class_lengths, filename):
    all_lens = np.concatenate([np.array(v) for v in class_lengths.values()])
    bins = np.linspace(0, np.percentile(all_lens, 99), 60)  # corta o 1% mais longo
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in sorted(class_lengths):
        ax.hist(class_lengths[label], bins=bins, alpha=0.6, label=name_of(label))
    ax.set_xlabel("Tokens por documento (após pré-processamento)")
    ax.set_ylabel("Documentos")
    ax.set_title("Histograma do tamanho dos documentos")
    ax.legend()
    save(fig, filename)


def plot_top_words(total, filename, n=30):
    words, counts = zip(*total.most_common(n))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(words[::-1], counts[::-1], color="#337ab7")
    ax.set_xlabel("Frequência")
    ax.set_title(f"Top {n} palavras mais frequentes")
    save(fig, filename)


def plot_zipf(total, filename):
    freqs = np.array(sorted(total.values(), reverse=True))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, len(freqs) + 1), freqs)
    ax.set_xlabel("Rank da palavra")
    ax.set_ylabel("Frequência")
    ax.set_title("Lei de Zipf (escala log-log)")
    save(fig, filename)


def plot_wordclouds(class_words, total, filename):
    panels = [("Corpus completo", total)] + [(name_of(l), class_words[l]) for l in sorted(class_words)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    for ax, (title, counter) in zip(axes, panels):
        wc = WordCloud(width=800, height=600, background_color="white", max_words=150)
        ax.imshow(wc.generate_from_frequencies(dict(counter.most_common(300))), interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")
    save(fig, filename)


def plot_distinctive_words(class_words, total, filename, top_n=15, vocab_size=5000):
    """Razão de log-chances (com suavização de Laplace) entre as duas classes."""
    a, b = sorted(class_words)[:2]
    vocab = [w for w, _ in total.most_common(vocab_size)]
    na, nb, V = sum(class_words[a].values()), sum(class_words[b].values()), len(vocab)
    score = {w: log((class_words[b][w] + 1) / (nb + V)) - log((class_words[a][w] + 1) / (na + V))
             for w in vocab}
    ranked = sorted(score.items(), key=lambda kv: kv[1])
    picked = ranked[:top_n] + ranked[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([w for w, _ in picked], [s for _, s in picked],
            color=["#d9534f"] * top_n + ["#5cb85c"] * top_n)
    ax.set_title(f"Palavras mais associadas a {name_of(a)} (←) e {name_of(b)} (→)")
    ax.set_xlabel("log-odds ratio")
    save(fig, filename)


def explore():
    """Estatísticas e gráficos; devolve a frequência das palavras do treino."""
    results = {name: scan(path) for name, path in PROCESSED.items()}

    for name, (cw, cl) in results.items():
        df, _, hapax = summarize(cw, cl)
        print(f"\n===== {name} =====")
        print(f"Número de classes: {len(cw)}")
        print(df.round(2).to_string(index=False))
        print(f"Palavras que aparecem uma única vez (hapax): {hapax:,}")
        plot_class_distribution(cl, f"classes_{name}.png")

    cw, cl = results["train"]
    _, total, _ = summarize(cw, cl)
    plot_length_hist(cl, "hist_tamanho_docs.png")
    plot_top_words(total, "top_palavras.png")
    plot_zipf(total, "zipf.png")
    plot_wordclouds(cw, total, "wordclouds.png")
    plot_distinctive_words(cw, total, "palavras_distintivas.png")
    return total


def read_chunks(path, chunk_size=CHUNK_SIZE):
    with open(path, encoding="utf-8") as f:
        while chunk := list(islice(f, chunk_size)):
            yield chunk


def vectorize(path, counter):
    """Bag-of-Words do arquivo inteiro, em blocos para limitar o uso de memória."""
    blocks, labels = [], []
    with tqdm(total=count_lines(path), desc=f"BoW {path.name}", unit="docs") as progress:
        for lines in read_chunks(path):
            pairs = [line.rstrip("\n").partition(" ") for line in lines]
            labels.extend(LABEL_IDS[label] for label, _, _ in pairs)
            blocks.append(counter.transform(text for _, _, text in pairs))
            progress.update(len(lines))
    return sp.vstack(blocks, format="csr"), np.array(labels, dtype=np.int8)


def to_tfidf(bow, tfidf):
    return sp.vstack([tfidf.transform(bow[i:i + CHUNK_SIZE]).astype(np.float32)
                      for i in range(0, bow.shape[0], CHUNK_SIZE)], format="csr")


def build_numeric_dataset(total):
    """BoW e TF-IDF de treino e teste. Vocabulário e IDF vêm só do treino (sem vazamento)."""
    vocab = [w for w, _ in total.most_common(VOCAB_SIZE)]
    # O texto já está tokenizado: basta separar por espaço, sem descartar tokens de 1 letra
    counter = CountVectorizer(vocabulary=vocab, tokenizer=str.split, token_pattern=None,
                              lowercase=False, dtype=np.int32)
    tfidf = TfidfTransformer(sublinear_tf=True)

    print(f"\nDataset numérico: {len(vocab):,} termos (os mais frequentes do treino)")
    for split, path in PROCESSED.items():
        bow, y = vectorize(path, counter)
        if split == "train":
            tfidf.fit(bow)
        sp.save_npz(ASSETS / f"{split}.bow.npz", bow, compressed=False)
        sp.save_npz(ASSETS / f"{split}.tfidf.npz", to_tfidf(bow, tfidf), compressed=False)
        np.save(ASSETS / f"{split}.labels.npy", y)
        print(f"  {split}: {bow.shape[0]:,} docs x {bow.shape[1]:,} termos, "
              f"densidade {100 * bow.nnz / (bow.shape[0] * bow.shape[1]):.3f}%")
    (ASSETS / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")


def main():
    build_numeric_dataset(explore())
    print(f"\nFiguras salvas em {OUT.resolve()}")
    print(f"Dataset numérico salvo em {ASSETS.resolve()}")


if __name__ == "__main__":
    main()
