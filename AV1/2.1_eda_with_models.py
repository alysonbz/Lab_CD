from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer

MODEL_INPUTS = Path("./assets/model_inputs")
OUT = Path("./outputs/eda_with_models")
OUT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = np.array(["negativo", "positivo"])
SEED = 42
SAMPLE_SIZE = 100_000
N_TOPICS = 10


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def load_sample():
    y = np.load(MODEL_INPUTS / "train.labels.npy")
    idx = np.random.default_rng(SEED).choice(len(y), min(SAMPLE_SIZE, len(y)), replace=False)
    X = sp.load_npz(MODEL_INPUTS / "train.tfidf.npz")[idx]
    terms = np.array((MODEL_INPUTS / "vocab.txt").read_text(encoding="utf-8").split("\n"))
    return X, CLASS_NAMES[y[idx]], terms


def top_terms(weights, terms, n=10):
    return ", ".join(terms[np.argsort(weights)[::-1][:n]])


def lsa(X):
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Z = Normalizer(copy=False).fit_transform(svd.fit_transform(X))
    summary = pd.DataFrame([{"documentos": X.shape[0], "termos": X.shape[1],
                             "componentes": svd.n_components,
                             "variancia_explicada": svd.explained_variance_ratio_.sum()}])
    summary.to_csv(OUT / "lsa.csv", index=False)
    print(f"\nAmostra TF-IDF: {X.shape[0]:,} docs x {X.shape[1]:,} termos")
    print(f"Variância explicada pelo SVD (100 comp.): {svd.explained_variance_ratio_.sum():.1%}")
    return Z


def tsne(Z, labels):
    sub = np.random.default_rng(SEED).choice(len(Z), size=min(5000, len(Z)), replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z[sub])
    points = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "classe": labels[sub]})
    points.to_csv(OUT / "tsne.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, group in points.groupby("classe"):
        ax.scatter(group.x, group.y, s=4, alpha=0.5, label=lab)
    ax.legend(markerscale=4)
    ax.set_title(f"t-SNE sobre TF-IDF + SVD ({len(sub):,} docs)")
    save(fig, "tsne.png")


def kmeans(X, Z, labels, terms):
    rows = []
    for k in range(2, 11):
        km = MiniBatchKMeans(k, random_state=SEED, n_init=3, batch_size=4096).fit(Z)
        rows.append({"k": k, "inercia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_, sample_size=10_000, random_state=SEED),
                     "ari": adjusted_rand_score(labels, km.labels_)})
    selection = pd.DataFrame(rows)
    selection.to_csv(OUT / "kmeans_selecao_k.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(selection.k, selection.inercia, marker="o")
    axes[0].set(title="Método do cotovelo", xlabel="k", ylabel="Inércia")
    axes[1].plot(selection.k, selection.silhouette, marker="o")
    axes[1].set(title="Silhouette", xlabel="k", ylabel="Score")
    save(fig, "kmeans_k.png")

    best_k = int(selection.k[selection.silhouette.idxmax()])
    km = MiniBatchKMeans(best_k, random_state=SEED, n_init=10, batch_size=4096).fit(Z)
    clusters = pd.crosstab(km.labels_, labels).add_prefix("docs_").rename_axis("cluster").rename_axis(None, axis=1)
    clusters["top_termos"] = [top_terms(np.asarray(X[km.labels_ == c].mean(axis=0)).ravel(), terms)
                              for c in clusters.index]
    clusters.to_csv(OUT / "kmeans_clusters.csv")
    final = pd.DataFrame([{"k": best_k, "inercia": km.inertia_,
                           "silhouette": silhouette_score(Z, km.labels_, sample_size=10_000, random_state=SEED),
                           "ari": adjusted_rand_score(labels, km.labels_)}])
    final.to_csv(OUT / "kmeans_final.csv", index=False)
    print(f"\n[K-Means] k escolhido por silhouette = {best_k}")
    print(f"ARI vs. rótulos reais: {final.ari[0]:.3f}")
    print(clusters.to_string())


def nmf_topics(X, labels, terms):
    nmf = NMF(n_components=N_TOPICS, init="nndsvd", random_state=SEED, max_iter=300)
    W = nmf.fit_transform(X)
    # Proporção dos documentos de cada classe cujo tópico dominante é t
    topics = (pd.crosstab(W.argmax(axis=1), labels, normalize="columns")
              .reindex(range(N_TOPICS), fill_value=0)
              .add_prefix("prop_").rename_axis("topico").rename_axis(None, axis=1))
    topics.insert(0, "top_termos", [top_terms(nmf.components_[t], terms) for t in topics.index])
    topics.to_csv(OUT / "nmf_topicos.csv")
    print(f"\n[NMF] {N_TOPICS} tópicos")
    print(topics.round(3).to_string())
    ax = topics.drop(columns="top_termos").plot(kind="bar", figsize=(8, 4))
    ax.set(title="Tópico dominante por classe", xlabel="Tópico", ylabel="Proporção dos docs da classe")
    save(ax.get_figure(), "nmf_topicos.png")


def main():
    X, labels, terms = load_sample()
    Z = lsa(X)
    tsne(Z, labels)
    kmeans(X, Z, labels, terms)
    nmf_topics(X, labels, terms)
    print(f"\nResultados salvos em {OUT.resolve()}")


if __name__ == "__main__":
    main()
