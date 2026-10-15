"""
Etapa 3 (Questão 4) - Análise de erros do modelo escolhido NA VALIDAÇÃO (melhor_config.json).
Usa as predições salvas; não treina nada. Descreve ONDE o modelo erra, para a discussão crítica.

Grava em resultados/: tabela_confusoes_mais_comuns.csv, tabela_erros_por_grupo.csv,
  tabela_relatedness_acertos_erros.csv, erros_modelo_escolhido.csv (exemplos para ler)
"""
import json

import numpy as np
import pandas as pd

from q4_utils import NOMES_BONITOS, PASTA_RESULTADOS, ROTULOS, carregar_teste, q3

prev = pd.read_csv(q3.exigir_arquivo(PASTA_RESULTADOS / "predicoes_teste.csv", "Rode antes a etapa 1."))
teste = carregar_teste()
assert prev["sentence_pair_id"].tolist() == teste["sentence_pair_id"].tolist()
with open(q3.PASTA_RESULTADOS / "melhor_config.json", encoding="utf-8") as f:
    info = json.load(f)
nome = info["melhor_geral_na_validacao"]
chave = f"{nome}|{info['configuracoes'][nome]['vetorizador']}"
teste = teste.assign(real=prev["y_real"].to_numpy(), previsto=prev[chave].to_numpy())
teste["acertou"] = teste["real"] == teste["previsto"]
print(f"Modelo analisado (escolhido na validação): {NOMES_BONITOS[nome]}  [{chave}]")
print(f"acurácia no teste: {teste['acertou'].mean():.3f}  ({int((~teste['acertou']).sum())} erros em {len(teste)} pares)")

# 1) tipos de erro mais comuns
erros = teste[~teste["acertou"]]
conf = (erros.groupby(["real", "previsto"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False))
conf["real"] = conf["real"].map(ROTULOS)
conf["previsto"] = conf["previsto"].map(ROTULOS)
conf["pct_dos_erros"] = (100 * conf["n"] / conf["n"].sum()).round(1)
conf.to_csv(PASTA_RESULTADOS / "tabela_confusoes_mais_comuns.csv", index=False)
print("\n=== 1. Tipos de erro (real -> previsto) ===")
print(conf.to_string(index=False))

# 2) desempenho por grupo de pares (atributos descritivos, calculados só do texto)
rel = q3.atributos_relacao(teste)
grupos = {
    "negação assimétrica": rel["neg_assimetrica"].map({0: "não", 1: "sim"}),
    "mesmos números nas duas frases": rel["numeros_iguais"].map({0: "não", 1: "sim"}),
    "sobreposição (Jaccard)": pd.qcut(rel["jaccard"].rank(method="first"), 3, labels=["baixa", "média", "alta"]),
}
linhas = []
for rotulo, serie in grupos.items():
    for valor, idx in teste.groupby(serie, observed=True).groups.items():
        g = teste.loc[idx]
        linhas.append(dict(grupo=rotulo, valor=str(valor), n=len(g), acuracia=g["acertou"].mean(),
                           pct_real_none=100 * (g["real"] == 0).mean(),
                           pct_previsto_none=100 * (g["previsto"] == 0).mean()))
t = pd.DataFrame(linhas).round(3)
t.to_csv(PASTA_RESULTADOS / "tabela_erros_por_grupo.csv", index=False)
print("\n=== 2. Acurácia por tipo de par ===")
print(t.to_string(index=False))

# 3) relatedness_score de acertos e erros (SÓ descritivo: nunca é feature)
r = (teste.groupby(["real", "acertou"])["relatedness_score"].agg(["count", "mean"]).round(2).reset_index())
r["real"] = r["real"].map(ROTULOS)
r.to_csv(PASTA_RESULTADOS / "tabela_relatedness_acertos_erros.csv", index=False)
print("\n=== 3. relatedness_score médio: pares acertados x errados, por classe real ===")
print(r.to_string(index=False))

# 4) exemplos para ler
partes = [g.sample(n=min(15, len(g)), random_state=q3.SEMENTE) for _, g in erros.groupby(["real", "previsto"])]
ex = pd.concat(partes)[["sentence_pair_id", "real", "previsto", "relatedness_score", "premise", "hypothesis"]]
ex = ex.assign(real=ex["real"].map(ROTULOS), previsto=ex["previsto"].map(ROTULOS))
ex.to_csv(PASTA_RESULTADOS / "erros_modelo_escolhido.csv", index=False)
print("\n=== 4. Exemplos de erro (5 de cada tipo; mais em erros_modelo_escolhido.csv) ===")
for (re_, pr_), g in ex.groupby(["real", "previsto"]):
    print(f"\n[real {re_} -> previsto {pr_}]")
    for _, e in g.head(5).iterrows():
        print(f"  P: {e['premise']}\n  H: {e['hypothesis']}   (score {e['relatedness_score']})")
print("\nLeia os exemplos antes de escrever a discussão: eles mostram se os erros são do modelo ou de rótulos discutíveis.")
