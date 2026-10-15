"""
Funções compartilhadas da Questão 4 (avaliação quantitativa, ASSIN ptbr).

Este arquivo NÃO é executado diretamente. Ele reaproveita o q3_utils.py (pasta questão-3)
para reconstruir os modelos e é o ÚNICO lugar do projeto que carrega o conjunto de TESTE.

  PythonProject/
  ├── questão-1/dados_processados/   <- entrada (test_processado.parquet)
  ├── questão-3/                     <- q3_utils.py, modelos/ e resultados/ da Q3
  └── questão-4/                     <- este arquivo
      ├── resultados/                <- tabelas (.csv), predições e tabelas em LaTeX
      └── figuras/                   <- gráficos (.png)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PASTA_QUESTAO = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_QUESTAO.parent
PASTA_Q3 = PASTA_PROJETO / "questão-3"
PASTA_RESULTADOS = PASTA_QUESTAO / "resultados"
PASTA_FIGURAS = PASTA_QUESTAO / "figuras"

sys.path.insert(0, str(PASTA_Q3))
import q3_utils as q3  # noqa: E402  (precisa vir depois do sys.path)

ROTULOS = q3.ROTULOS
CLASSES = [0, 1, 2]
NOMES_BONITOS = {"NaiveBayes": "Naive Bayes", "RegressaoLogistica": "Regressão Logística",
                 "SVM": "SVM linear", "MLP": "MLP"}
NOMES_VET = {"tfidf": "TF-IDF", "count": "CountVectorizer"}
SEMENTE = 42
N_BOOTSTRAP = 2000


def carregar_teste():
    """Único ponto do projeto que lê o conjunto de teste."""
    return pd.read_parquet(q3.exigir_arquivo(
        q3.PASTA_DADOS / "test_processado.parquet", "Rode antes as etapas 2 e 3 da Questão 1."))


def f1_macro_rapido(y, p):
    """F1 macro via matriz de confusão (usado no bootstrap). F1 = 2TP / (n_previstos + n_reais)."""
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)))


def bootstrap_f1(y, previsoes, n_boot=N_BOOTSTRAP, semente=SEMENTE):
    """F1 macro em n_boot reamostragens do teste (com reposição). As MESMAS reamostragens
    valem para todos os modelos, o que permite comparar dois modelos de forma pareada."""
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    return {k: np.array([f1_macro_rapido(y[i], p[i]) for i in idx]) for k, p in previsoes.items()}


def ic95(valores):
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def num(x, casas=3):
    """Formata com vírgula decimal (padrão do relatório)."""
    return f"{x:.{casas}f}".replace(".", ",")
