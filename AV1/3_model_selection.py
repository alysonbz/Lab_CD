from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODEL_INPUTS = Path("./assets/model_inputs")
TEST_TEXTS = {  # mesma ordem de linhas de test.labels.npy
    "texto": Path("./assets/test.ft.txt"),
    "texto_processado": Path("./assets/test.processed.txt"),
}
OUT = Path("./outputs/model_selection")
OUT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = np.array(["negativo", "positivo"])
SEED = 42
N_JOBS = 2  # cada fit paralelo copia o treino (~3 GB de RAM)
TOP_WORDS = 15
N_ERRORS = 10
REPRESENTATIONS = ["bow", "tfidf"]
COMPARED_METRICS = ["f1_validacao", "acuracia", "precisao", "recall", "f1", "tempo_treino_s"]
MODELS = {
    "naive_bayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0,]}),
    "svm": (LinearSVC(random_state=SEED), {"C": [0.01, 0.1, 1.0]}),
}


def load(rep, split):
    return sp.load_npz(MODEL_INPUTS / f"{split}.{rep}.npz")


def load_labels(split):
    return np.load(MODEL_INPUTS / f"{split}.labels.npy")


def read_lines(path, indices):
    wanted = set(indices)
    with open(path, encoding="utf-8") as f:
        found = {i: line.rstrip("\n").partition(" ")[2] for i, line in enumerate(f) if i in wanted}
    return [found[i] for i in indices]


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


# Peso > 0 favorece positivo
def feature_weights(model):
    if hasattr(model, "coef_"):  # SVM
        return model.coef_[0]
    return model.feature_log_prob_[1] - model.feature_log_prob_[0]  # Naive Bayes


# Margem > 0 prevê positivo; |margem| mede a confiança
def decision_margin(model, X):
    if hasattr(model, "decision_function"):  # SVM
        return model.decision_function(X)
    log_proba = model.predict_log_proba(X)  # Naive Bayes
    return log_proba[:, 1] - log_proba[:, 0]


def confident_errors(y_test, pred, margin, n=N_ERRORS):
    tables = []
    for real, kind in [(0, "falso_positivo"), (1, "falso_negativo")]:
        idx = np.flatnonzero((y_test == real) & (pred != real))
        idx = idx[np.argsort(-np.abs(margin[idx]))[:n]]
        tables.append(pd.DataFrame({"tipo": kind, "doc": idx, "margem": margin[idx]}))
    return pd.concat(tables)


def plot_confusion(name, preds, y_test):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4))
    tables = []
    for ax, (rep, pred) in zip(axes, preds.items()):
        disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASS_NAMES, ax=ax,
                                                       colorbar=False, values_format=",d")
        ax.set_title(f"{name} ({rep})")
        cm = pd.DataFrame(disp.confusion_matrix, columns=[f"previsto_{c}" for c in CLASS_NAMES])
        cm.insert(0, "real", CLASS_NAMES)
        cm.insert(0, "representacao", rep)
        tables.append(cm)
    save(fig, f"{name}_matriz_confusao.png")
    pd.concat(tables).to_csv(OUT / f"{name}_matriz_confusao.csv", index=False)


def plot_top_words(name, models, terms, n=TOP_WORDS):
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
    tables = []
    for ax, (rep, model) in zip(axes, models.items()):
        weights = feature_weights(model)
        order = np.argsort(weights)
        idx = np.concatenate([order[:n], order[-n:]])
        table = pd.DataFrame({"representacao": rep, "classe": np.repeat(CLASS_NAMES, n),
                              "palavra": terms[idx], "peso": weights[idx]})
        ax.barh(table.palavra, table.peso, color=["#d9534f"] * n + ["#5cb85c"] * n)
        ax.set(title=f"{name} ({rep})", xlabel="peso a favor de positivo")
        tables.append(table)
    save(fig, f"{name}_palavras_importantes.png")
    pd.concat(tables).to_csv(OUT / f"{name}_palavras_importantes.csv", index=False)


def plot_errors_by_length(preds, y_test, lengths):
    bins = pd.qcut(lengths, 5, duplicates="drop")
    tables = []
    for name, by_rep in preds.items():
        for rep, pred in by_rep.items():
            errors = pd.Series(pred != y_test).groupby(bins, observed=True).agg(["size", "mean"])
            tables.append(errors.rename(columns={"size": "docs", "mean": "taxa_erro"})
                          .rename_axis("faixa_tokens").reset_index()
                          .assign(modelo=name, representacao=rep))
    table = pd.concat(tables)[["modelo", "representacao", "faixa_tokens", "docs", "taxa_erro"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, rep), group in table.groupby(["modelo", "representacao"]):
        ax.plot(group.faixa_tokens.astype(str), group.taxa_erro, marker="o", label=f"{name} ({rep})")
    ax.set(title="Taxa de erro por tamanho do documento", xlabel="Tokens no vocabulário (quintis)",
           ylabel="Taxa de erro")
    ax.legend()
    save(fig, "erros_por_tamanho.png")
    table.to_csv(OUT / "erros_por_tamanho.csv", index=False)


def compare_representations(comparison):
    results = comparison.set_index(["modelo", "representacao"])
    table = pd.DataFrame([
        {"modelo": name, "metrica": metric,
         "bow": results.loc[(name, "bow"), metric], "tfidf": results.loc[(name, "tfidf"), metric]}
        for name in MODELS for metric in COMPARED_METRICS
    ])
    table["tfidf_menos_bow"] = table.tfidf - table.bow
    table.to_csv(OUT / "comparacao_bow_tfidf.csv", index=False)
    print("\n===== BoW x TF-IDF =====")
    print(table.round(4).to_string(index=False))


def save_error_examples(examples, y_test):
    docs = examples.doc.to_numpy()
    examples = examples.assign(real=CLASS_NAMES[y_test[docs]], previsto=CLASS_NAMES[1 - y_test[docs]])
    for column, path in TEST_TEXTS.items():
        examples[column] = read_lines(path, docs.tolist())
    columns = ["modelo", "representacao", "tipo", "doc", "real", "previsto", "margem", *TEST_TEXTS]
    examples[columns].to_csv(OUT / "erros_exemplos.csv", index=False)


def supervised():
    y_train, y_test = load_labels("train"), load_labels("test")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    validation, report = defaultdict(list), defaultdict(list)
    preds, models = defaultdict(dict), defaultdict(dict)
    summary, errors = [], []
    for rep in REPRESENTATIONS:
        X_train, X_test = load(rep, "train"), load(rep, "test")
        for name, (estimator, grid) in MODELS.items():
            # refit=True (padrão): best_estimator_ é re-treinado no treino completo
            search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=N_JOBS)
            search.fit(X_train, y_train)
            res = search.cv_results_
            validation[name].append(pd.DataFrame({
                "representacao": rep,
                **{p: res[f"param_{p}"] for p in grid},
                "f1_val_media": res["mean_test_score"],
                "f1_val_desvio": res["std_test_score"],
            }))

            model = models[name][rep] = search.best_estimator_
            pred = preds[name][rep] = model.predict(X_test)
            errors.append(confident_errors(y_test, pred, decision_margin(model, X_test))
                          .assign(modelo=name, representacao=rep))

            p, r, f, s = precision_recall_fscore_support(y_test, pred)
            pm, rm, fm, _ = precision_recall_fscore_support(y_test, pred, average="macro")
            report[name].append(pd.DataFrame({
                "representacao": rep, "classe": [*CLASS_NAMES, "macro"],
                "precisao": [*p, pm], "recall": [*r, rm], "f1": [*f, fm], "suporte": [*s, s.sum()],
            }))
            summary.append({
                "modelo": name, "representacao": rep,
                "hiperparametros": ", ".join(f"{k}={v}" for k, v in search.best_params_.items()),
                "f1_validacao": search.best_score_, "acuracia": accuracy_score(y_test, pred),
                "precisao": pm, "recall": rm, "f1": fm, "tempo_treino_s": search.refit_time_,
            })
            print(f"[{name} / {rep}] {summary[-1]['hiperparametros']}: "
                  f"F1 teste = {fm:.4f} (treino em {search.refit_time_:.1f}s)")
        del X_train, X_test

    terms = np.array((MODEL_INPUTS / "vocab.txt").read_text(encoding="utf-8").split("\n"))
    for name in MODELS:
        pd.concat(validation[name]).to_csv(OUT / f"{name}_validacao.csv", index=False)
        pd.concat(report[name]).to_csv(OUT / f"{name}.csv", index=False)
        plot_confusion(name, preds[name], y_test)
        plot_top_words(name, models[name], terms)
    comparison = pd.DataFrame(summary).sort_values("f1", ascending=False)
    comparison.to_csv(OUT / "comparacao_modelos.csv", index=False)
    print("\n===== Comparação no teste =====")
    print(comparison.round(4).to_string(index=False))
    compare_representations(comparison)

    lengths = np.asarray(load("bow", "test").sum(axis=1)).ravel()
    plot_errors_by_length(preds, y_test, lengths)
    save_error_examples(pd.concat(errors), y_test)


def main():
    supervised()
    print(f"\nResultados salvos em {OUT.resolve()}")


if __name__ == "__main__":
    main()
