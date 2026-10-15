"""
Etapa 4 (Questão 4) - Verificação. Cada check imprime OK ou FALHA.
Rode depois das etapas 1, 2 e 3.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from q4_utils import (CLASSES, NOMES_BONITOS, PASTA_FIGURAS, PASTA_RESULTADOS, ROTULOS,
                      carregar_teste, f1_macro_rapido, q3)

falhas = 0


def checar(descricao, condicao):
    global falhas
    print(("OK    " if condicao else "FALHA ") + descricao)
    if not condicao:
        falhas += 1


teste = carregar_teste()
y = teste["entailment_judgment"].to_numpy()
prev = pd.read_csv(q3.exigir_arquivo(PASTA_RESULTADOS / "predicoes_teste.csv", "Rode antes a etapa 1."))
cfg = pd.read_csv(PASTA_RESULTADOS / "configuracoes_avaliadas.csv")
cfg["escolhida_na_validacao"] = cfg["escolhida_na_validacao"].astype(str) == "True"
info = json.load(open(q3.PASTA_RESULTADOS / "melhor_config.json", encoding="utf-8"))

print("--- predições ---")
checar("mesmo número de linhas do teste", len(prev) == len(teste))
checar("sentence_pair_id na mesma ordem do teste", prev["sentence_pair_id"].tolist() == teste["sentence_pair_id"].tolist())
checar("y_real igual aos rótulos do teste", prev["y_real"].tolist() == y.tolist())
checar("8 configurações avaliadas (4 modelos x 2 vetorizadores)", len(cfg) == 8 and set(cfg["modelo"]) == set(q3.MODELOS))
checar("previsões só têm as classes 0, 1 e 2", all(set(prev[c].unique()) <= {0, 1, 2} for c in cfg["chave"]))
checar("sem nulos nas predições", int(prev.isna().sum().sum()) == 0)

print("\n--- seleção feita na validação, não no teste ---")
for m in q3.MODELOS:
    esc = cfg[(cfg["modelo"] == m) & cfg["escolhida_na_validacao"]]
    checar(f"{m}: exatamente 1 configuração marcada como escolhida", len(esc) == 1)
    checar(f"{m}: a escolhida coincide com melhor_config.json da Q3",
           len(esc) == 1 and esc.iloc[0]["vetorizador"] == info["configuracoes"][m]["vetorizador"]
           and esc.iloc[0]["estrategia"] == info["configuracoes"][m]["estrategia"])
    if len(esc) == 1:
        checar(f"{m}: F1 macro da validação reproduz o da Q3",
               abs(esc.iloc[0]["f1_macro_validacao"] - info["configuracoes"][m]["f1_macro_validacao"]) < 1e-9)

print("\n--- coerência com os modelos salvos na Q3 ---")
for m in q3.MODELOS:
    pacote = joblib.load(q3.exigir_arquivo(q3.PASTA_MODELOS / f"{m}.joblib", "Rode antes a Questão 3."))
    esc = cfg[(cfg["modelo"] == m) & cfg["escolhida_na_validacao"]].iloc[0]
    checar(f"{m}: predições no teste = as do modelo salvo da Q3",
           np.array_equal(pacote.prever(teste), prev[esc["chave"]].to_numpy()))

print("\n--- métricas e matrizes ---")
t1 = pd.read_csv(PASTA_RESULTADOS / "tabela_modelos_teste.csv")
checar("baseline: acurácia = proporção de NONE no teste",
       abs(t1.iloc[0]["acuracia"] - (y == 0).mean()) < 1e-3)
for m in q3.MODELOS:
    chave = cfg[(cfg["modelo"] == m) & cfg["escolhida_na_validacao"]].iloc[0]["chave"]
    p = prev[chave].to_numpy()
    linha = t1[t1["modelo"] == NOMES_BONITOS[m]].iloc[0]
    checar(f"{m}: acurácia da tabela = sklearn", abs(linha["acuracia"] - accuracy_score(y, p)) < 1e-3)
    checar(f"{m}: F1 macro da tabela = sklearn",
           abs(linha["f1_macro"] - f1_score(y, p, average="macro", labels=CLASSES)) < 1e-3)
    checar(f"{m}: F1 macro rápido (bootstrap) = sklearn", abs(f1_macro_rapido(y, p) - f1_score(y, p, average="macro")) < 1e-9)
    checar(f"{m}: IC 95% contém o F1 macro", linha["f1_ic95_inf"] - 1e-3 <= linha["f1_macro"] <= linha["f1_ic95_sup"] + 1e-3)
    cm = confusion_matrix(y, p, labels=CLASSES)
    checar(f"{m}: matriz de confusão soma {len(y)} e linhas = suporte real",
           cm.sum() == len(y) and cm.sum(axis=1).tolist() == [int((y == c).sum()) for c in CLASSES])
mc = pd.read_csv(PASTA_RESULTADOS / "matrizes_confusao.csv")
checar("matrizes_confusao.csv: cada modelo soma o tamanho do teste",
       all(g["n"].sum() == len(y) for _, g in mc.groupby("modelo")))
t3 = pd.read_csv(PASTA_RESULTADOS / "tabela_tfidf_vs_count_teste.csv")
for _, r in t3.iterrows():
    checar(f"{r['modelo']}: Δ F1 = F1 TF-IDF - F1 Count", abs(r["dif_f1"] - (r["f1_tfidf"] - r["f1_count"])) < 2e-3)
    checar(f"{r['modelo']}: IC do Δ contém o Δ observado", r["dif_ic95_inf"] - 1e-3 <= r["dif_f1"] <= r["dif_ic95_sup"] + 1e-3)

print("\n--- arquivos gerados ---")
for arq in ["fig_matrizes_confusao.png", "fig_comparacao_modelos.png", "fig_tfidf_vs_count.png"]:
    p = PASTA_FIGURAS / arq
    checar(f"figura {arq} existe e não está vazia", p.is_file() and p.stat().st_size > 0)
tex = (PASTA_RESULTADOS / "tabelas_latex.tex").read_text(encoding="utf-8")
checar("tabelas_latex.tex tem 3 tabelas", tex.count(r"\begin{table}") == 3 and tex.count(r"\end{table}") == 3)
for arq in ["tabela_confusoes_mais_comuns.csv", "tabela_erros_por_grupo.csv",
            "tabela_relatedness_acertos_erros.csv", "erros_modelo_escolhido.csv"]:
    checar(f"{arq} existe", (PASTA_RESULTADOS / arq).is_file())
er = pd.read_csv(PASTA_RESULTADOS / "erros_modelo_escolhido.csv")
checar("erros_modelo_escolhido.csv: todo exemplo é de fato um erro", bool((er["real"] != er["previsto"]).all()))

print("\nRESULTADO:", "tudo OK" if falhas == 0 else f"{falhas} falha(s)")
