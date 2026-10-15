"""
Utilidades compartilhadas da Questão 2 (análise exploratória).
"""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # só grava arquivos .png; não abre janelas
import matplotlib.pyplot as plt
import pandas as pd

PASTA_QUESTAO = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_QUESTAO.parent
PASTA_DADOS_Q1 = PASTA_PROJETO / "questão-1" / "dados_processados"
PASTA_FIG = PASTA_QUESTAO / "figuras"
PASTA_RES = PASTA_QUESTAO / "resultados"
PASTA_NUM = PASTA_QUESTAO / "dados_numericos"

SPLITS = ["train", "validation", "test"]
ROTULOS = {0: "NONE", 1: "ENTAILMENT", 2: "PARAPHRASE"}
CORES = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
NEGACOES = {"não", "nem", "nunca", "jamais", "nenhum", "nenhuma", "ninguém", "nada", "sem"}
# palavras funcionais muito frequentes (protegidas na Questão 1): ficam nas tabelas,
# mas são omitidas das nuvens, onde ocupariam quase todo o espaço
FUNCIONAIS = {"não", "nem", "nunca", "jamais", "nenhum", "nenhuma", "ninguém", "nada", "sem",
              "mais", "menos", "muito", "pouco", "todo", "quase", "apenas", "somente", "só",
              "mas", "porém", "contudo", "embora", "ainda", "já", "antes", "depois", "até", "contra"}
SEMENTE = 42

plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})


def criar_pastas():
    for p in (PASTA_FIG, PASTA_RES, PASTA_NUM):
        p.mkdir(parents=True, exist_ok=True)


def exigir_arquivo(caminho, dica=""):
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}\n{dica}".strip())
    return caminho


def carregar(split):
    """Lê o Parquet processado na Questão 1."""
    caminho = exigir_arquivo(
        PASTA_DADOS_Q1 / f"{split}_processado.parquet",
        "Rode antes as etapas da Questão 1 (principalmente a etapa2-preprocessar.py).",
    )
    return pd.read_parquet(caminho)


def contar(serie):
    """Conta palavras de uma coluna de textos já separados por espaço."""
    c = Counter()
    for texto in serie:
        c.update(texto.split())
    return c


def e_palavra(w):
    """True para palavras 'de verdade' (descarta números, símbolos como r$ e 3,8)."""
    return w.isalpha()


def salvar_fig(fig, nome):
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_FIG / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figura] {caminho.name}")


def salvar_csv(df, nome, **kw):
    PASTA_RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(PASTA_RES / nome, encoding="utf-8-sig", **kw)  # utf-8-sig: abre certo no Excel
    print(f"  [tabela] {nome}")