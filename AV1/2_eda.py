"""Análise exploratória do corpus (formato fastText, saída do pré-processamento).

Dependências: pip install numpy pandas matplotlib scikit-learn wordcloud tqdm
"""
import random
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from wordcloud import WordCloud

PROCESSED = {
    "train": "./assets/train.processed.txt",
    "test": "./assets/test.processed.txt",
}
LABEL_NAMES = {"__label__1": "negativo", "__label__2": "positivo"}
SAMPLE_SIZE = 100_000
SEED = 42
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
    n_docs = count_lines(path)
    p = min(1.0, SAMPLE_SIZE / n_docs)
    rng = random.Random(SEED)
    class_words = defaultdict(Counter)
    class_lengths = defaultdict(list)
    sample_labels, sample_texts = [], []

    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=n_docs, desc=Path(path).name, unit="docs"):
            label, _, text = line.rstrip("\n").partition(" ")
            tokens = text.split()
            class_words[label].update(tokens)
            class_lengths[label].append(len(tokens))
            if tokens and rng.random() < p:
                sample_labels.append(label)
                sample_texts.append(text)
    return class_words, class_lengths, sample_labels, sample_texts


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


def unsupervised(texts, labels):
    labels = np.array([name_of(l) for l in labels])
    rng = np.random.default_rng(SEED)

    tfidf = TfidfVectorizer(max_features=20_000, min_df=5, max_df=0.5, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())
    print(f"\nMatriz TF-IDF: {X.shape[0]:,} docs x {X.shape[1]:,} termos "
          f"(densidade {100 * X.nnz / (X.shape[0] * X.shape[1]):.3f}%)")

    # Redução de dimensionalidade (LSA) usada por clusterização e visualização
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Z = Normalizer(copy=False).fit_transform(svd.fit_transform(X))
    print(f"Variância explicada pelo SVD (100 comp.): {svd.explained_variance_ratio_.sum():.1%}")

    # --- Aplicação 1: redução de dimensionalidade + visualização (LSA + t-SNE) ---
    idx = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z[idx])
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in np.unique(labels):
        m = labels[idx] == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, label=lab)
    ax.legend(markerscale=4)
    ax.set_title("t-SNE sobre TF-IDF + SVD (5 mil docs)")
    save(fig, "app1_tsne.png")

    # --- Aplicação 2: clusterização (MiniBatchKMeans) ---
    ks = list(range(2, 11))
    inertias, sils = [], []
    for k in ks:
        km = MiniBatchKMeans(k, random_state=SEED, n_init=3, batch_size=4096).fit(Z)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Z, km.labels_, sample_size=10_000, random_state=SEED))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set(title="Método do cotovelo", xlabel="k", ylabel="Inércia")
    axes[1].plot(ks, sils, marker="o")
    axes[1].set(title="Silhouette", xlabel="k", ylabel="Score")
    save(fig, "app2_kmeans_k.png")

    best_k = ks[int(np.argmax(sils))]
    km = MiniBatchKMeans(best_k, random_state=SEED, n_init=10, batch_size=4096).fit(Z)
    print(f"\n[K-Means] k escolhido por silhouette = {best_k}")
    print(f"ARI vs. rótulos reais: {adjusted_rand_score(labels, km.labels_):.3f}")
    print(pd.crosstab(km.labels_, labels, rownames=["cluster"]))
    for c in range(best_k):
        centroid = np.asarray(X[km.labels_ == c].mean(axis=0)).ravel()
        print(f"  cluster {c}: " + ", ".join(terms[centroid.argsort()[::-1][:10]]))

    # --- Aplicação 3: modelagem de tópicos (NMF sobre TF-IDF) ---
    n_topics = 10
    nmf = NMF(n_components=n_topics, init="nndsvd", random_state=SEED, max_iter=300)
    W = nmf.fit_transform(X)
    print(f"\n[NMF] {n_topics} tópicos")
    for t, comp in enumerate(nmf.components_):
        print(f"  tópico {t}: " + ", ".join(terms[comp.argsort()[::-1][:10]]))
    dist = pd.crosstab(labels, W.argmax(axis=1), normalize="index")
    ax = dist.T.plot(kind="bar", figsize=(8, 4))
    ax.set(title="Tópico dominante por classe", xlabel="Tópico", ylabel="Proporção dos docs da classe")
    save(ax.get_figure(), "app3_nmf_topicos.png")


def main():
    results = {name: scan(path) for name, path in PROCESSED.items()}

    for name, (cw, cl, _, _) in results.items():
        df, _, hapax = summarize(cw, cl)
        print(f"\n===== {name} =====")
        print(df.round(2).to_string(index=False))
        print(f"Palavras que aparecem uma única vez (hapax): {hapax:,}")
        plot_class_distribution(cl, f"classes_{name}.png")

    cw, cl, sample_labels, sample_texts = results["train"]
    _, total, _ = summarize(cw, cl)
    plot_length_hist(cl, "hist_tamanho_docs.png")
    plot_top_words(total, "top_palavras.png")
    plot_zipf(total, "zipf.png")
    plot_wordclouds(cw, total, "wordclouds.png")
    plot_distinctive_words(cw, total, "palavras_distintivas.png")

    unsupervised(sample_texts, sample_labels)
    print(f"\nFiguras salvas em {OUT.resolve()}")


if __name__ == "__main__":
    main()