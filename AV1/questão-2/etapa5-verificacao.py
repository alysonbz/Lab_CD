"""
Etapa 5 - Verificação da Questão 2. Cada check imprime OK ou FALHA.
"""
import json
import numpy as np
import pandas as pd
from scipy import sparse

from eda_utils import (PASTA_FIG, PASTA_NUM, PASTA_RES, ROTULOS, SPLITS, carregar,
                       exigir_arquivo)

falhas = 0


def checar(descricao, condicao):
    global falhas
    print(("OK    " if condicao else "FALHA ") + descricao)
    if not condicao:
        falhas += 1


print("--- arquivos gerados ---")
figuras = ["fig_distribuicao_classes", "fig_comprimento_por_classe", "fig_relatedness_por_classe",
           "fig_top_palavras", "fig_top_palavras_por_classe", "fig_zipf", "fig_negacao_por_classe",
           "fig_sobreposicao", "fig_nuvens_frequencia", "fig_nuvens_distintivas",
           "fig_reducao_dimensionalidade", "fig_kmeans", "fig_topicos_por_classe", "fig_similaridade"]
for f in figuras:
    p = PASTA_FIG / f"{f}.png"
    checar(f"figura {f}.png existe e não está vazia", p.is_file() and p.stat().st_size > 5_000)
tabelas = ["tamanho_corpus", "distribuicao_classes", "top_palavras", "top_palavras_por_classe",
           "palavras_distintivas", "negacao_por_classe", "sobreposicao_por_classe",
           "atributos_relacao_por_classe", "kmeans_relacao_composicao", "kmeans_conteudo_termos",
           "topicos_nmf", "topicos_por_classe", "pares_atipicos"]
for t in tabelas:
    checar(f"tabela {t}.csv existe", (PASTA_RES / f"{t}.csv").is_file())

print("\n--- consistência com os dados processados ---")
tam = pd.read_csv(PASTA_RES / "tamanho_corpus.csv").set_index("conjunto")
dist = pd.read_csv(PASTA_RES / "distribuicao_classes.csv", index_col=0)
for s in SPLITS:
    d = carregar(s)
    checar(f"{s}: nº de pares na tabela = {len(d)}", int(tam.loc[s, "pares"]) == len(d))
    contagem = d["entailment_judgment"].value_counts().sort_index().tolist()
    checar(f"{s}: contagem por classe na tabela = {contagem}", dist[f"n_{s}"].tolist() == contagem)
checar("soma dos conjuntos = linha 'total'", int(tam.loc[SPLITS, "pares"].sum()) == int(tam.loc["total", "pares"]))
checar("3 classes em todos os conjuntos", (tam["n_classes"] == len(ROTULOS)).all())

print("\n--- dataset numérico ---")
vocab = pd.read_csv(PASTA_NUM / "vocabulario_tfidf.csv", encoding="utf-8-sig")["termo"].tolist()
tr = carregar("train")
palavras_treino = set(w for s in tr["premise_lemas_sem_sw"] for w in s.split()) | \
                  set(w for s in tr["hypothesis_lemas_sem_sw"] for w in s.split())
checar("vocabulário TF-IDF só contém palavras que existem no treino (sem vazamento)", set(vocab) <= palavras_treino)
for s in SPLITS:
    d = carregar(s)
    Xt = sparse.load_npz(PASTA_NUM / f"X_tfidf_{s}.npz")
    Xr = pd.read_parquet(PASTA_NUM / f"X_rel_{s}.parquet")
    meta = pd.read_parquet(PASTA_NUM / f"meta_{s}.parquet")
    checar(f"{s}: linhas de TF-IDF, atributos e meta batem com os dados ({len(d)})",
           Xt.shape[0] == len(Xr) == len(meta) == len(d))
    checar(f"{s}: TF-IDF tem {len(vocab)} colunas", Xt.shape[1] == len(vocab))
    checar(f"{s}: sem NaN/inf nos atributos de relação", bool(np.isfinite(Xr.to_numpy()).all()))
    checar(f"{s}: rótulos em meta iguais aos dos dados", meta["entailment_judgment"].tolist() == d["entailment_judgment"].tolist())
    checar(f"{s}: atributos não contêm rótulo, relatedness_score nem id",
           not ({"entailment_judgment", "relatedness_score", "sentence_pair_id"} & set(Xr.columns)))
    norma = np.sqrt(np.asarray(Xt.multiply(Xt).sum(axis=1)).ravel())
    checar(f"{s}: linhas do TF-IDF com norma 1 (ou 0 se nenhuma palavra do vocabulário)",
           bool(np.all(np.isclose(norma, 1) | np.isclose(norma, 0))))
    checar(f"{s}: cos_tfidf entre 0 e 1", bool(Xr["cos_tfidf"].between(-1e-9, 1 + 1e-9).all()))
    checar(f"{s}: jaccard entre 0 e 1", bool(Xr["jaccard"].between(0, 1).all()))

print("\nRESULTADO:", "tudo OK" if falhas == 0 else f"{falhas} falha(s)")