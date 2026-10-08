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
VOCAB_SIZE = 50_000
CHUNK_SIZE = 100_000
MODEL_INPUTS = ASSETS / "model_inputs"
MODEL_INPUTS.mkdir(exist_ok=True)
RAW_TRAIN = ASSETS / "train.ft.txt"
EXAMPLE = MODEL_INPUTS / "example"
EXAMPLE.mkdir(exist_ok=True)
EXAMPLES_PER_CLASS = 3
EXAMPLE_TOKENS = range(8, 21)  # nº de tokens dos docs da amostra legível
OUT = Path("./outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)


def name_of(label):
    return LABEL_NAMES.get(label, label)


def count_lines(path):
    with open(path, "rb") as f:
        return sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))


def save(fig, filename, table):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    table.to_csv((OUT / filename).with_suffix(".csv"), index=False)


def hapax(counter):
    return sum(1 for c in counter.values() if c == 1)


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
            "máx": lens.max(), "hapax": hapax(class_words[label]),
        })
    rows.append({
        "classe": "TOTAL", "documentos": n_total, "% docs": 100.0,
        "tokens": int(all_lens.sum()), "vocabulário": len(total),
        "média tokens/doc": all_lens.mean(), "mediana": np.median(all_lens),
        "máx": all_lens.max(), "hapax": hapax(total),
    })
    return pd.DataFrame(rows), total


def plot_class_distribution(class_lengths, filename):
    labels = sorted(class_lengths)
    counts = [len(class_lengths[l]) for l in labels]
    table = pd.DataFrame({"classe": [name_of(l) for l in labels], "documentos": counts,
                          "% docs": [100 * c / sum(counts) for c in counts]})
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(table.classe, counts, color=["#d9534f", "#5cb85c"][:len(labels)])
    for bar, c, pct in zip(bars, counts, table["% docs"]):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n({pct:.1f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Documentos")
    ax.set_title("Distribuição de classes")
    ax.set_ylim(0, max(counts) * 1.2)
    save(fig, filename, table)


def plot_length_hist(class_lengths, filename):
    all_lens = np.concatenate([np.array(v) for v in class_lengths.values()])
    bins = np.linspace(0, np.percentile(all_lens, 99), 60)
    table = pd.DataFrame({"tokens_de": bins[:-1], "tokens_ate": bins[1:]})
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in sorted(class_lengths):
        table[name_of(label)] = np.histogram(class_lengths[label], bins=bins)[0]
        ax.hist(class_lengths[label], bins=bins, alpha=0.6, label=name_of(label))
    ax.set_xlabel("Tokens por documento (após pré-processamento)")
    ax.set_ylabel("Documentos")
    ax.set_title("Histograma do tamanho dos documentos")
    ax.legend()
    save(fig, filename, table)


def plot_top_words(total, filename, n=30):
    table = pd.DataFrame(total.most_common(n), columns=["palavra", "frequencia"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(table.palavra[::-1], table.frequencia[::-1], color="#337ab7")
    ax.set_xlabel("Frequência")
    ax.set_title(f"Top {n} palavras mais frequentes")
    save(fig, filename, table)


def plot_zipf(total, filename):
    table = pd.DataFrame(total.most_common(), columns=["palavra", "frequencia"])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(table["rank"], table.frequencia)
    ax.set_xlabel("Rank da palavra")
    ax.set_ylabel("Frequência")
    ax.set_title("Lei de Zipf (escala log-log)")
    save(fig, filename, table)


def plot_wordclouds(class_words, total, filename, max_words=150):
    panels = [("Corpus completo", total)] + [(name_of(l), class_words[l]) for l in sorted(class_words)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    rows = []
    for ax, (title, counter) in zip(axes, panels):
        top = counter.most_common(max_words)
        wc = WordCloud(width=800, height=600, background_color="white", max_words=max_words)
        ax.imshow(wc.generate_from_frequencies(dict(top)), interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")
        rows += [{"painel": title, "palavra": w, "frequencia": c} for w, c in top]
    save(fig, filename, pd.DataFrame(rows))


def plot_distinctive_words(class_words, total, filename, top_n=15, vocab_size=5000):
    a, b = sorted(class_words)[:2]
    vocab = [w for w, _ in total.most_common(vocab_size)]
    na, nb, V = sum(class_words[a].values()), sum(class_words[b].values()), len(vocab)
    score = {w: log((class_words[b][w] + 1) / (nb + V)) - log((class_words[a][w] + 1) / (na + V))
             for w in vocab}
    ranked = sorted(score.items(), key=lambda kv: kv[1])
    table = pd.DataFrame(ranked[:top_n] + ranked[-top_n:], columns=["palavra", "log_odds"])
    table.insert(0, "classe", [name_of(a)] * top_n + [name_of(b)] * top_n)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(table.palavra, table.log_odds, color=["#d9534f"] * top_n + ["#5cb85c"] * top_n)
    ax.set_title(f"Palavras mais associadas a {name_of(a)} (←) e {name_of(b)} (→)")
    ax.set_xlabel("log-odds ratio")
    save(fig, filename, table)


def explore():
    results = {name: scan(path) for name, path in PROCESSED.items()}

    for name, (cw, cl) in results.items():
        df, _ = summarize(cw, cl)
        df.to_csv(OUT / f"resumo_{name}.csv", index=False)
        print(f"\n===== {name} =====")
        print(f"Número de classes: {len(cw)}")
        print(df.round(2).to_string(index=False))
        plot_class_distribution(cl, f"classes_{name}.png")

    cw, cl = results["train"]
    _, total = summarize(cw, cl)
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


def save_examples(counter, tfidf, vocab):
    picked, per_class = {}, Counter()
    with open(PROCESSED["train"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            label, _, text = line.rstrip("\n").partition(" ")
            if per_class[label] < EXAMPLES_PER_CLASS and len(text.split()) in EXAMPLE_TOKENS:
                picked[i] = (label, text)
                per_class[label] += 1
            if len(picked) == EXAMPLES_PER_CLASS * len(LABEL_NAMES):
                break
    with open(RAW_TRAIN, encoding="utf-8") as f:
        raw = [line.rstrip("\n").partition(" ")[2] for line in islice(f, max(picked) + 1)]

    in_vocab = set(vocab)
    docs = pd.DataFrame([{
        "doc": i, "classe": name_of(label), "rotulo": LABEL_IDS[label],
        "texto_original": raw[i], "texto_processado": text,
        "fora_do_vocabulario": " ".join(w for w in text.split() if w not in in_vocab),
    } for i, (label, text) in picked.items()])
    docs.to_csv(EXAMPLE / "documentos.csv", index=False)

    terms = np.array(vocab)
    bow = counter.transform(docs.texto_processado)
    weights = tfidf.transform(bow)
    used = np.unique(bow.indices)
    index = pd.MultiIndex.from_arrays([docs.doc, docs.rotulo], names=["doc", "rotulo"])
    for filename, matrix in [("bow.csv", bow), ("tfidf.csv", weights)]:
        table = pd.DataFrame(matrix[:, used].toarray(), index=index, columns=terms[used])
        table.round(4).to_csv(EXAMPLE / filename)

    coo = bow.tocoo()
    pd.DataFrame({
        "doc": docs.doc.to_numpy()[coo.row], "coluna": coo.col, "termo": terms[coo.col],
        "contagem": coo.data, "tf_sublinear": 1 + np.log(coo.data), "idf": tfidf.idf_[coo.col],
        "tfidf": np.asarray(weights[coo.row, coo.col]).ravel(),
    }).round(4).to_csv(EXAMPLE / "esparso.csv", index=False)


def build_numeric_dataset(total):
    # Vocabulário e IDF vêm só do treino (sem vazamento do teste)
    vocab = [w for w, _ in total.most_common(VOCAB_SIZE)]
    # Texto já tokenizado: str.split não descarta tokens de 1 letra, como o token_pattern padrão faria
    counter = CountVectorizer(vocabulary=vocab, tokenizer=str.split, token_pattern=None,
                              lowercase=False, dtype=np.int32)
    tfidf = TfidfTransformer(sublinear_tf=True)

    print(f"\nDataset numérico: {len(vocab):,} termos (os mais frequentes do treino)")
    rows = []
    for split, path in PROCESSED.items():
        bow, y = vectorize(path, counter)
        if split == "train":
            tfidf.fit(bow)
        sp.save_npz(MODEL_INPUTS / f"{split}.bow.npz", bow, compressed=False)
        sp.save_npz(MODEL_INPUTS / f"{split}.tfidf.npz", to_tfidf(bow, tfidf), compressed=False)
        np.save(MODEL_INPUTS / f"{split}.labels.npy", y)
        rows.append({"split": split, "documentos": bow.shape[0], "termos": bow.shape[1],
                     "nao_zeros": bow.nnz, "densidade_%": 100 * bow.nnz / (bow.shape[0] * bow.shape[1])})
        print(f"  {split}: {bow.shape[0]:,} docs x {bow.shape[1]:,} termos, "
              f"densidade {rows[-1]['densidade_%']:.3f}%")
    (MODEL_INPUTS / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "dataset_numerico.csv", index=False)
    save_examples(counter, tfidf, vocab)


def main():
    build_numeric_dataset(explore())
    print(f"\nFiguras e tabelas salvas em {OUT.resolve()}")
    print(f"Dataset numérico salvo em {MODEL_INPUTS.resolve()} (amostra legível em {EXAMPLE.name}/)")


if __name__ == "__main__":
    main()
