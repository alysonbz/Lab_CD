"""
Funções compartilhadas da Questão 3 (classificação de texto, ASSIN ptbr).

Este arquivo NÃO é executado diretamente: é importado pelas etapas 1, 2 e 3
(e será importado pela Questão 4, para avaliar os modelos salvos no TESTE).

Regra de ouro: aqui só se carrega TREINO e VALIDAÇÃO. O teste não é lido
em nenhuma etapa da Questão 3; ele fica reservado para a Questão 4.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------------
# Caminhos (partem da localização DESTE arquivo, como nas Questões 1 e 2)
#   PythonProject/
#   ├── questão-1/dados_processados/   <- ENTRADA (gerada pela Questão 1)
#   └── questão-3/                     <- este arquivo
#       ├── resultados/                <- tabelas (.csv) e melhor_config.json
#       └── modelos/                   <- modelos treinados (.joblib)
# ----------------------------------------------------------------------
PASTA_QUESTAO = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_QUESTAO.parent
PASTA_DADOS = PASTA_PROJETO / "questão-1" / "dados_processados"
PASTA_RESULTADOS = PASTA_QUESTAO / "resultados"
PASTA_MODELOS = PASTA_QUESTAO / "modelos"

SEMENTE = 42
ROTULOS = {0: "NONE", 1: "ENTAILMENT", 2: "PARAPHRASE"}
NEGACOES = {"não", "nem", "nunca", "jamais", "nenhum", "nenhuma", "ninguém", "nada", "sem"}

VETORIZADORES = ["tfidf", "count"]
NGRAMAS = [(1, 1), (1, 2)]
ESTRATEGIAS = ["nenhuma", "pesos", "oversampling"]
MODELOS = ["NaiveBayes", "RegressaoLogistica", "SVM", "MLP"]

# Hiperparâmetros testados (escolhidos pela VALIDAÇÃO, nunca pelo teste)
GRADE = {
    "NaiveBayes": [{"alpha": a} for a in (0.1, 0.5, 1.0)],
    "RegressaoLogistica": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "SVM": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "MLP": [{"alpha": a} for a in (1e-4, 1e-2)],
}
# MLP do scikit-learn não aceita pesos de classe; Naive Bayes só aceita números >= 0
# (por isso não recebe os atributos de relação, que têm valores negativos).
ESTRATEGIAS_POR_MODELO = {
    "NaiveBayes": ESTRATEGIAS,
    "RegressaoLogistica": ESTRATEGIAS,
    "SVM": ESTRATEGIAS,
    "MLP": ["nenhuma", "oversampling"],
}
ATRIBUTOS_RELACAO_POR_MODELO = {
    "NaiveBayes": [False],
    "RegressaoLogistica": [False, True],
    "SVM": [False, True],
    "MLP": [False, True],
}

NOMES_REL = ["jaccard", "h_em_p", "p_em_h", "n_lemas_p", "n_lemas_h", "dif_tamanho",
             "neg_p", "neg_h", "neg_assimetrica", "numeros_iguais", "numeros_novos_h"]


# ----------------------------------------------------------------------
# Leitura dos dados
# ----------------------------------------------------------------------
def exigir_arquivo(caminho, dica=""):
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}\n{dica}".strip())
    return caminho


def _ler(nome):
    return pd.read_parquet(exigir_arquivo(
        PASTA_DADOS / nome, "Rode antes as etapas 2 e 3 da Questão 1 (pasta questão-1)."))


def carregar_dados():
    """Devolve (treino, validacao, treino_oversampled, pesos). NÃO carrega o teste."""
    treino = _ler("train_processado.parquet")
    validacao = _ler("validation_processado.parquet")
    over = _ler("train_oversampled.parquet")
    with open(exigir_arquivo(PASTA_DADOS / "class_weights.json",
                             "Rode antes a etapa 3 da Questão 1."), encoding="utf-8") as f:
        pesos = {int(k): float(v) for k, v in json.load(f).items()}
    return treino, validacao, over, pesos


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def texto_par(df):
    """Texto do par para BoW/TF-IDF: lemas sem stopwords (negações protegidas) de
    premissa + hipótese. Não usa relatedness_score nem sentence_pair_id."""
    return (df["premise_lemas_sem_sw"] + " " + df["hypothesis_lemas_sem_sw"]).tolist()


def _numeros(texto):
    return set(re.findall(r"\d+(?:[.,]\d+)?", texto))


def atributos_relacao(df):
    """11 atributos que descrevem a RELAÇÃO entre premissa e hipótese (a Q2 mostrou que o
    sinal está aí). Usam só as colunas de texto processado, sem rótulo e sem relatedness_score."""
    linhas = []
    for p, h, pl, hl, pt, ht in zip(df["premise_lemas_sem_sw"], df["hypothesis_lemas_sem_sw"],
                                    df["premise_lemas"], df["hypothesis_lemas"],
                                    df["premise_tokens"], df["hypothesis_tokens"]):
        sp, sh = set(p.split()), set(h.split())
        inter, uniao = len(sp & sh), len(sp | sh)
        neg_p = int(bool(NEGACOES & set(pl.split())))
        neg_h = int(bool(NEGACOES & set(hl.split())))
        num_p, num_h = _numeros(pt), _numeros(ht)
        n_p, n_h = len(p.split()), len(h.split())
        linhas.append([
            inter / uniao if uniao else 0.0,        # jaccard
            inter / len(sh) if sh else 0.0,         # fração da hipótese contida na premissa
            inter / len(sp) if sp else 0.0,         # fração da premissa contida na hipótese
            n_p, n_h, n_p - n_h,
            neg_p, neg_h, int(neg_p != neg_h),      # negação assimétrica
            int(num_p == num_h),                    # mesmos números nas duas frases
            len(num_h - num_p),                     # números da hipótese que não estão na premissa
        ])
    return pd.DataFrame(linhas, columns=NOMES_REL, index=df.index)


def criar_vetorizador(tipo, ngrama):
    # token_pattern aceita tokens de 1 caractere (números como "8" importam nesta tarefa).
    # TF-IDF e Count usam a MESMA configuração, para a comparação ser justa.
    comum = dict(ngram_range=ngrama, min_df=2, lowercase=False, token_pattern=r"(?u)\b\w+\b")
    if tipo == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **comum)
    if tipo == "count":
        return CountVectorizer(**comum)
    raise ValueError(tipo)


class Representacao:
    """Vetorizador (+ padronização dos atributos de relação), ajustados SOMENTE no treino
    original (sem duplicatas do oversampling). Validação, teste e oversampling só são transformados."""

    def __init__(self, vetorizador, ngrama, usa_rel):
        self.vetorizador, self.ngrama, self.usa_rel = vetorizador, tuple(ngrama), bool(usa_rel)
        self.vec = None
        self.escala = None

    def ajustar(self, treino):
        self.vec = criar_vetorizador(self.vetorizador, self.ngrama).fit(texto_par(treino))
        if self.usa_rel:
            self.escala = StandardScaler().fit(atributos_relacao(treino).values)
        return self

    def transformar(self, df):
        X = self.vec.transform(texto_par(df))
        if self.usa_rel:
            R = self.escala.transform(atributos_relacao(df).values)
            X = sparse.hstack([X, sparse.csr_matrix(R)]).tocsr()
        return X


# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
def criar_modelo(nome, params, estrategia, pesos):
    cw = pesos if estrategia == "pesos" else None
    if nome == "NaiveBayes":
        return MultinomialNB(**params)
    if nome == "RegressaoLogistica":
        return LogisticRegression(max_iter=3000, class_weight=cw, **params)
    if nome == "SVM":
        return LinearSVC(class_weight=cw, max_iter=20000, random_state=SEMENTE, **params)
    if nome == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=SEMENTE, **params)
    raise ValueError(nome)


def ajustar_modelo(modelo, nome, X, y, estrategia, pesos):
    """Ajusta o modelo. Retorna True se houve aviso de não-convergência."""
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        if nome == "NaiveBayes" and estrategia == "pesos":
            # MultinomialNB não tem class_weight, mas aceita sample_weight: mesmo efeito
            modelo.fit(X, y, sample_weight=np.array([pesos[int(c)] for c in y]))
        else:
            modelo.fit(X, y)
    return any(issubclass(a.category, ConvergenceWarning) for a in avisos)


def metricas(y_real, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_real, y_pred, labels=[0, 1, 2],
                                                 average=None, zero_division=0)
    return {
        "acuracia": accuracy_score(y_real, y_pred),
        "precisao_macro": float(p.mean()), "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "f1_NONE": float(f[0]), "f1_ENTAILMENT": float(f[1]), "f1_PARAPHRASE": float(f[2]),
        "recall_PARAPHRASE": float(r[2]),
    }


class ModeloTreinado:
    """Pacote salvo em modelos/*.joblib: representação + modelo + configuração.
    A Questão 4 só precisa de:  pacote.prever(df_teste)."""

    def __init__(self, nome, config, representacao, modelo):
        self.nome, self.config = nome, config
        self.representacao, self.modelo = representacao, modelo

    def prever(self, df):
        return self.modelo.predict(self.representacao.transformar(df))


def parse_ngrama(texto):
    a, b = str(texto).split("-")
    return (int(a), int(b))


def formata_ngrama(ng):
    return f"{ng[0]}-{ng[1]}"
