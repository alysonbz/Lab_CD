"""
Etapa 1 - Tamanho do corpus, número de classes e distribuição de classes.
Entrada : questão-1/dados_processados/*.parquet
Saídas  : resultados/tamanho_corpus.csv, resultados/distribuicao_classes.csv,
          figuras/fig_distribuicao_classes.png, fig_comprimento_por_classe.png,
          fig_relatedness_por_classe.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eda_utils import (CORES, ROTULOS, SPLITS, carregar, contar, criar_pastas,
                       salvar_csv, salvar_fig)

criar_pastas()
dfs = {s: carregar(s) for s in SPLITS}
dfs_todos = pd.concat(dfs.values(), ignore_index=True)

# ----------------------------------------------------------------------
# 1. Tamanho do corpus
# ----------------------------------------------------------------------
def resumo(nome, d):
    tok_p, tok_h = d["premise_tokens"], d["hypothesis_tokens"]
    c_tok = contar(pd.concat([tok_p, tok_h]))
    c_lem = contar(pd.concat([d["premise_lemas"], d["hypothesis_lemas"]]))
    return {
        "conjunto": nome,
        "pares": len(d),
        "frases_unicas": pd.concat([d["premise"], d["hypothesis"]]).nunique(),
        "tokens_total": sum(c_tok.values()),
        "vocabulario_tokens": len(c_tok),
        "vocabulario_lemas": len(c_lem),
        "palavras_que_aparecem_1x": sum(1 for n in c_tok.values() if n == 1),
        "tokens_por_premissa": round(tok_p.str.split().str.len().mean(), 2),
        "tokens_por_hipotese": round(tok_h.str.split().str.len().mean(), 2),
        "n_classes": d["entailment_judgment"].nunique(),
    }

tabela = pd.DataFrame([resumo(s, d) for s, d in dfs.items()] + [resumo("total", dfs_todos)])
print("=== 1. Tamanho do corpus ===")
print(tabela.to_string(index=False))
salvar_csv(tabela, "tamanho_corpus.csv", index=False)

print("\nClasses:", {c: n for c, n in ROTULOS.items()}, f"-> {len(ROTULOS)} classes")

# ----------------------------------------------------------------------
# 2. Distribuição de classes
# ----------------------------------------------------------------------
cont = pd.DataFrame({s: d["entailment_judgment"].value_counts().sort_index() for s, d in dfs.items()})
cont["total"] = cont.sum(axis=1)
prop = (cont / cont.sum() * 100).round(1)
saida = pd.concat([cont.add_prefix("n_"), prop.add_prefix("pct_")], axis=1)
saida.index = [f"{c} - {ROTULOS[c]}" for c in saida.index]
print("\n=== 2. Distribuição de classes (n e %) ===")
print(saida.to_string())
salvar_csv(saida, "distribuicao_classes.csv")

maior, menor = cont["train"].max(), cont["train"].min()
print(f"\nRazão de desbalanceamento no treino (maior/menor classe): {maior / menor:.1f}:1")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
barras = a1.bar([ROTULOS[c] for c in cont.index], cont["train"], color=[CORES[c] for c in cont.index])
for b, n, p in zip(barras, cont["train"], prop["train"]):
    a1.text(b.get_x() + b.get_width() / 2, n, f"{n}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
a1.set_ylim(0, cont["train"].max() * 1.18)
a1.set_title("Treino: exemplos por classe"); a1.set_ylabel("nº de pares")
larg = 0.25
for i, s in enumerate(SPLITS):
    a2.bar(np.arange(3) + i * larg, prop[s], larg, label=s)
a2.set_xticks(np.arange(3) + larg); a2.set_xticklabels([ROTULOS[c] for c in cont.index])
a2.set_title("Proporção de cada classe por conjunto"); a2.set_ylabel("%"); a2.legend()
fig.tight_layout(); salvar_fig(fig, "fig_distribuicao_classes.png")

# ----------------------------------------------------------------------
# 3. Comprimento das frases por classe (treino)
# ----------------------------------------------------------------------
tr = dfs["train"].copy()
tr["n_p"] = tr["premise_tokens"].str.split().str.len()
tr["n_h"] = tr["hypothesis_tokens"].str.split().str.len()
tr["dif"] = tr["n_h"] - tr["n_p"]

print("\n=== 3. Comprimento (tokens) por classe, treino ===")
print(tr.groupby("entailment_judgment")[["n_p", "n_h", "dif"]].agg(["mean", "std"]).round(2).to_string())

fig, eixos = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, col, titulo in zip(eixos, ["n_p", "n_h", "dif"],
                           ["Premissa: nº de tokens", "Hipótese: nº de tokens", "Hipótese − premissa (tokens)"]):
    bins = np.arange(tr[col].min() - 0.5, tr[col].max() + 1.5, 1)
    for c in sorted(ROTULOS):
        ax.hist(tr.loc[tr["entailment_judgment"] == c, col], bins=bins, density=True,
                histtype="step", linewidth=1.8, color=CORES[c], label=ROTULOS[c])
    ax.set_title(titulo); ax.set_ylabel("densidade")
eixos[0].legend()
fig.tight_layout(); salvar_fig(fig, "fig_comprimento_por_classe.png")

# ----------------------------------------------------------------------
# 4. relatedness_score por classe (treino)  -- só análise, NÃO é feature
# ----------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
bins = np.arange(0.75, 5.5, 0.5)
for c in sorted(ROTULOS):
    a1.hist(tr.loc[tr["entailment_judgment"] == c, "relatedness_score"], bins=bins,
            alpha=0.6, color=CORES[c], label=ROTULOS[c])
a1.set_title("relatedness_score (treino)"); a1.set_xlabel("pontuação de similaridade"); a1.set_ylabel("nº de pares")
a1.legend()
a2.boxplot([tr.loc[tr["entailment_judgment"] == c, "relatedness_score"] for c in sorted(ROTULOS)])
a2.set_xticklabels([ROTULOS[c] for c in sorted(ROTULOS)])
a2.set_title("relatedness_score por classe"); a2.set_ylabel("pontuação")
fig.tight_layout(); salvar_fig(fig, "fig_relatedness_por_classe.png")
print("\nrelatedness_score por classe:")
print(tr.groupby("entailment_judgment")["relatedness_score"].describe().round(2).to_string())