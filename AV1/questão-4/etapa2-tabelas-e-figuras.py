"""
Etapa 2 (Questão 4) - Métricas, matrizes de confusão, TF-IDF x CountVectorizer,
intervalos de confiança (bootstrap), figuras e tabelas em LaTeX.
Usa SÓ as predições salvas pela etapa 1 (não treina nem lê modelos).

Grava em resultados/: tabela_modelos_teste.csv, tabela_por_classe_teste.csv,
  tabela_tfidf_vs_count_teste.csv, matrizes_confusao.csv, tabelas_latex.tex
Grava em figuras/: fig_matrizes_confusao.png, fig_comparacao_modelos.png, fig_tfidf_vs_count.png
"""
import matplotlib
matplotlib.use("Agg")  # só salva arquivos; nenhuma janela abre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from q4_utils import (CLASSES, NOMES_BONITOS, NOMES_VET, PASTA_FIGURAS, PASTA_RESULTADOS, ROTULOS,
                      bootstrap_f1, ic95, num, q3)

PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
prev = pd.read_csv(q3.exigir_arquivo(PASTA_RESULTADOS / "predicoes_teste.csv", "Rode antes a etapa 1."))
cfg = pd.read_csv(PASTA_RESULTADOS / "configuracoes_avaliadas.csv")
cfg["atributos_relacao"] = cfg["atributos_relacao"].astype(str) == "True"
y = prev["y_real"].to_numpy()
P = {c: prev[c].to_numpy() for c in cfg["chave"]}
MODELOS = q3.MODELOS
finais = {r["modelo"]: r["chave"] for _, r in cfg[cfg["escolhida_na_validacao"].astype(str) == "True"].iterrows()}
f1_val = dict(zip(cfg["chave"], cfg["f1_macro_validacao"]))

# baseline: sempre NONE
P["Baseline"] = np.zeros_like(y)
boot = bootstrap_f1(y, P)


def linha_metricas(chave):
    m = q3.metricas(y, P[chave])
    lo, hi = ic95(boot[chave])
    return {**m, "f1_ic95_inf": lo, "f1_ic95_sup": hi}


# ---------------------------------------------------------------- tabela 1: modelos finais
linhas = [dict(modelo="Baseline (sempre NONE)", vetorizador="-", ngrama="-", atributos_relacao="-",
               estrategia="-", f1_macro_validacao=np.nan, **linha_metricas("Baseline"))]
for m in MODELOS:
    c = cfg[cfg["chave"] == finais[m]].iloc[0]
    linhas.append(dict(modelo=NOMES_BONITOS[m], vetorizador=NOMES_VET[c["vetorizador"]], ngrama=c["ngrama"],
                       atributos_relacao="sim" if c["atributos_relacao"] else "não",
                       estrategia=c["estrategia"], f1_macro_validacao=f1_val[c["chave"]],
                       **linha_metricas(c["chave"])))
t1 = pd.DataFrame(linhas)
t1["gap_val_teste"] = t1["f1_macro_validacao"] - t1["f1_macro"]
t1.round(3).to_csv(PASTA_RESULTADOS / "tabela_modelos_teste.csv", index=False)
print("=== 1. Modelos finais no TESTE (configuração escolhida na validação) ===")
print(t1.round(3).to_string(index=False))
print("\n(IC 95% do F1 macro por bootstrap com 2000 reamostragens do teste; "
      "gap = F1 validação - F1 teste)")

# ---------------------------------------------------------------- tabela 2: por classe
linhas = []
for rotulo, chave in [("Baseline (sempre NONE)", "Baseline")] + [(NOMES_BONITOS[m], finais[m]) for m in MODELOS]:
    p, r, f, s = precision_recall_fscore_support(y, P[chave], labels=CLASSES, zero_division=0)
    for i, c in enumerate(CLASSES):
        linhas.append(dict(modelo=rotulo, classe=ROTULOS[c], precision=p[i], recall=r[i], f1=f[i], suporte=int(s[i])))
t2 = pd.DataFrame(linhas)
t2.round(3).to_csv(PASTA_RESULTADOS / "tabela_por_classe_teste.csv", index=False)
print("\n=== 2. Precision, recall e F1 por classe (teste) ===")
print(t2.round(3).to_string(index=False))

# ---------------------------------------------------------------- matrizes de confusão
cms, longo = {}, []
for m in MODELOS:
    cm = confusion_matrix(y, P[finais[m]], labels=CLASSES)
    cms[m] = cm
    for i in CLASSES:
        for j in CLASSES:
            longo.append(dict(modelo=NOMES_BONITOS[m], real=ROTULOS[i], previsto=ROTULOS[j], n=int(cm[i, j])))
pd.DataFrame(longo).to_csv(PASTA_RESULTADOS / "matrizes_confusao.csv", index=False)
print("\n=== 3. Matrizes de confusão (linhas = real, colunas = previsto) ===")
for m in MODELOS:
    print(f"\n{NOMES_BONITOS[m]}")
    print(pd.DataFrame(cms[m], index=[f"real {ROTULOS[c]}" for c in CLASSES],
                       columns=[f"prev {ROTULOS[c]}" for c in CLASSES]).to_string())

fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, m in zip(axs.ravel(), MODELOS):
    cm = cms[m]
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in CLASSES:
        for j in CLASSES:
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks(CLASSES, [ROTULOS[c] for c in CLASSES], fontsize=9)
    ax.set_yticks(CLASSES, [ROTULOS[c] for c in CLASSES], fontsize=9)
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    ax.set_title(f"{NOMES_BONITOS[m]}  (F1 macro {q3.metricas(y, P[finais[m]])['f1_macro']:.3f})")
fig.suptitle("Matrizes de confusão no teste (% = proporção da classe real)")
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "fig_matrizes_confusao.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- comparação dos modelos
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
rotulos_m = [NOMES_BONITOS[m] for m in MODELOS]
f1s = np.array([q3.metricas(y, P[finais[m]])["f1_macro"] for m in MODELOS])
ics = np.array([ic95(boot[finais[m]]) for m in MODELOS])
a1.bar(rotulos_m, f1s, yerr=[f1s - ics[:, 0], ics[:, 1] - f1s], capsize=5, color="#4C78A8")
a1.axhline(t1.loc[0, "f1_macro"], color="gray", ls="--", label="sempre NONE")
a1.set_ylabel("F1 macro (teste)"); a1.set_ylim(0, 1); a1.legend()
a1.set_title("F1 macro com IC 95%"); a1.tick_params(axis="x", labelsize=8)
larg = 0.2
for k, c in enumerate(CLASSES):
    vals = [q3.metricas(y, P[finais[m]])[f"f1_{ROTULOS[c]}"] for m in MODELOS]
    a2.bar(np.arange(len(MODELOS)) + (k - 1) * larg, vals, larg, label=ROTULOS[c])
a2.set_xticks(range(len(MODELOS)), rotulos_m, fontsize=8)
a2.set_ylabel("F1 por classe (teste)"); a2.set_ylim(0, 1); a2.legend(); a2.set_title("F1 por classe")
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "fig_comparacao_modelos.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- TF-IDF x CountVectorizer
linhas = []
for m in MODELOS:
    a, b = f"{m}|tfidf", f"{m}|count"
    ma, mb = q3.metricas(y, P[a]), q3.metricas(y, P[b])
    d = boot[a] - boot[b]
    lo, hi = ic95(d)
    linhas.append(dict(modelo=NOMES_BONITOS[m], f1_tfidf=ma["f1_macro"], f1_count=mb["f1_macro"],
                       dif_f1=ma["f1_macro"] - mb["f1_macro"], dif_ic95_inf=lo, dif_ic95_sup=hi,
                       diferenca_significativa="sim" if (lo > 0 or hi < 0) else "não",
                       f1_paraphrase_tfidf=ma["f1_PARAPHRASE"], f1_paraphrase_count=mb["f1_PARAPHRASE"],
                       acuracia_tfidf=ma["acuracia"], acuracia_count=mb["acuracia"]))
t3 = pd.DataFrame(linhas)
t3.round(3).to_csv(PASTA_RESULTADOS / "tabela_tfidf_vs_count_teste.csv", index=False)
print("\n=== 4. TF-IDF x CountVectorizer no teste (cada um com sua melhor config da validação) ===")
print(t3.round(3).to_string(index=False))
print("(dif = TF-IDF - Count; IC 95% da diferença por bootstrap pareado)")

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(MODELOS))
for k, (vet, cor) in enumerate([("tfidf", "#4C78A8"), ("count", "#F58518")]):
    v = np.array([q3.metricas(y, P[f"{m}|{vet}"])["f1_macro"] for m in MODELOS])
    ic = np.array([ic95(boot[f"{m}|{vet}"]) for m in MODELOS])
    ax.bar(x + (k - 0.5) * 0.35, v, 0.35, yerr=[v - ic[:, 0], ic[:, 1] - v], capsize=4,
           label=NOMES_VET[vet], color=cor)
ax.set_xticks(x, [NOMES_BONITOS[m] for m in MODELOS]); ax.set_ylim(0, 1)
ax.set_ylabel("F1 macro (teste)"); ax.legend(); ax.set_title("TF-IDF x CountVectorizer")
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "fig_tfidf_vs_count.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- tabelas em LaTeX
def tex_tabela1():
    l = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Modelo & Configuração & Acurácia & \textit{Precision} & \textit{Recall} & F1 macro & IC 95\% & F1 valid. \\",
         r"\midrule"]
    for _, r in t1.iterrows():
        if r["modelo"].startswith("Baseline"):
            conf, fv = "--", "--"
        else:
            conf = f"{r['vetorizador']}, {r['ngrama']}, relação: {r['atributos_relacao']}, {r['estrategia']}"
            fv = num(r["f1_macro_validacao"])
        nome = r["modelo"].replace("Baseline (sempre NONE)", "\\textit{Baseline} (sempre \\textit{none})")
        l.append(f"{nome} & {conf} & {num(r['acuracia'])} & {num(r['precisao_macro'])} & {num(r['recall_macro'])} "
                 f"& {num(r['f1_macro'])} & [{num(r['f1_ic95_inf'])}; {num(r['f1_ic95_sup'])}] & {fv} \\\\")
    l += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Desempenho no teste dos modelos finais (configuração escolhida na validação). "
          r"\textit{Precision}, \textit{recall} e F1 são médias macro das três classes. "
          r"IC 95\% obtido por \textit{bootstrap}.}", r"\label{tab:q4modelos}", r"\end{table}"]
    return "\n".join(l)


def tex_tabela2():
    l = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Modelo & Classe & \textit{Precision} & \textit{Recall} & F1 & Suporte \\", r"\midrule"]
    for i, (nome, g) in enumerate(t2.groupby("modelo", sort=False)):
        if i:
            l.append(r"\midrule")
        nome = nome.replace("Baseline (sempre NONE)", "\\textit{Baseline}")
        for k, (_, r) in enumerate(g.iterrows()):
            l.append(f"{nome if k == 0 else ''} & \\textit{{{r['classe'].lower()}}} & {num(r['precision'])} "
                     f"& {num(r['recall'])} & {num(r['f1'])} & {r['suporte']} \\\\")
    l += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textit{Precision}, \textit{recall} e F1 por classe no teste.}",
          r"\label{tab:q4classes}", r"\end{table}"]
    return "\n".join(l)


def tex_tabela3():
    l = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Modelo & F1 TF-IDF & F1 Count & $\Delta$ & IC 95\% de $\Delta$ & F1$_{par}$ TF-IDF & F1$_{par}$ Count \\",
         r"\midrule"]
    for _, r in t3.iterrows():
        l.append(f"{r['modelo']} & {num(r['f1_tfidf'])} & {num(r['f1_count'])} & {num(r['dif_f1'])} "
                 f"& [{num(r['dif_ic95_inf'])}; {num(r['dif_ic95_sup'])}] "
                 f"& {num(r['f1_paraphrase_tfidf'])} & {num(r['f1_paraphrase_count'])} \\\\")
    l += [r"\bottomrule", r"\end{tabular}",
          r"\caption{TF-IDF contra \textit{CountVectorizer} no teste (F1 macro e F1 da classe \textit{paraphrase}). "
          r"Cada vetorizador usa sua melhor configuração na validação; $\Delta$ = TF-IDF $-$ Count. "
          r"Se o IC 95\% de $\Delta$ contém zero, a diferença não é conclusiva.}",
          r"\label{tab:q4vetorizadores}", r"\end{table}"]
    return "\n".join(l)


(PASTA_RESULTADOS / "tabelas_latex.tex").write_text(
    "% Gerado por etapa2-tabelas-e-figuras.py (requer booktabs e graphicx)\n\n"
    + tex_tabela1() + "\n\n" + tex_tabela2() + "\n\n" + tex_tabela3() + "\n", encoding="utf-8")
print("\n[salvo] resultados/tabelas_latex.tex  e figuras/fig_*.png")
