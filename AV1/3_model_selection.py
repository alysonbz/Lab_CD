"""Classificação sobre o dataset numérico gerado por 2_eda.py (rode-o antes).

Naive Bayes e SVM linear, cada um com BoW e com TF-IDF.
Hiperparâmetros escolhidos por validação cruzada numa amostra do treino;
a melhor configuração é re-treinada no treino completo e avaliada no teste.

Tabelas (CSV) e figuras em outputs/models/.

Dependências: pip install numpy pandas matplotlib scipy scikit-learn
"""
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ASSETS = Path("./assets")
OUT = Path("./outputs/models")
OUT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = np.array(["negativo", "positivo"])  # índices usados em *.labels.npy
SEED = 42
CV_SAMPLE = 200_000  # docs usados na validação cruzada dos hiperparâmetros
REPRESENTATIONS = ["bow", "tfidf"]
MODELS = {
    "naive_bayes": (MultinomialNB(), {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}),
    "svm": (LinearSVC(random_state=SEED), {"C": [0.001, 0.01, 0.1, 1.0]}),
}


def load(rep, split):
    return sp.load_npz(ASSETS / f"{split}.{rep}.npz")


def load_labels(split):
    return np.load(ASSETS / f"{split}.labels.npy")


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def plot_confusion(name, preds, y_test):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4))
    for ax, (rep, pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASS_NAMES, ax=ax,
                                                colorbar=False, values_format=",d")
        ax.set_title(f"{name} ({rep})")
    save(fig, f"{name}_matriz_confusao.png")


def supervised():
    y_train, y_test = load_labels("train"), load_labels("test")
    cv_idx = np.random.default_rng(SEED).choice(len(y_train), min(CV_SAMPLE, len(y_train)), replace=False)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    validation, report, preds, summary = defaultdict(list), defaultdict(list), defaultdict(dict), []
    for rep in REPRESENTATIONS:
        X_train, X_test = load(rep, "train"), load(rep, "test")
        for name, (estimator, grid) in MODELS.items():
            search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
            search.fit(X_train[cv_idx], y_train[cv_idx])
            res = search.cv_results_
            validation[name].append(pd.DataFrame({
                "representacao": rep,
                **{p: res[f"param_{p}"] for p in grid},
                "f1_val_media": res["mean_test_score"],
                "f1_val_desvio": res["std_test_score"],
            }))

            model = clone(estimator).set_params(**search.best_params_)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            pred = preds[name][rep] = model.predict(X_test)

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
                "precisao": pm, "recall": rm, "f1": fm, "tempo_treino_s": train_time,
            })
            print(f"[{name} / {rep}] {summary[-1]['hiperparametros']}: "
                  f"F1 teste = {fm:.4f} (treino em {train_time:.1f}s)")
        del X_train, X_test

    for name in MODELS:
        pd.concat(validation[name]).to_csv(OUT / f"{name}_validacao.csv", index=False)
        pd.concat(report[name]).to_csv(OUT / f"{name}.csv", index=False)
        plot_confusion(name, preds[name], y_test)
    comparison = pd.DataFrame(summary).sort_values("f1", ascending=False)
    comparison.to_csv(OUT / "comparacao_modelos.csv", index=False)
    print("\n===== Comparação no teste =====")
    print(comparison.round(4).to_string(index=False))


def main():
    supervised()
    print(f"\nResultados salvos em {OUT.resolve()}")


if __name__ == "__main__":
    main()
