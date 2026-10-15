"""
Etapa 2 - Frequência de palavras e visualizações (treino).
Usa o TREINO para não "espiar" validação/teste.
 - tokens        : lei de Zipf (vocabulário completo)
 - lemas_sem_sw  : palavras de conteúdo (top palavras, distintivas, nuvens)
Saídas: resultados/top_palavras*.csv, palavras_distintivas.csv, negacao_por_classe.csv,
        sobreposicao_por_classe.csv; figuras/fig_top_palavras*.png, fig_zipf.png,
        fig_negacao_por_classe.png, fig_sobreposicao.png, fig_nuvens*.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from eda_utils import (CORES, FUNCIONAIS, NEGACOES, ROTULOS, carregar, contar, criar_pastas,
                       e_palavra, salvar_csv, salvar_fig)

criar_pastas()
tr = carregar("train")
y = tr["entailment_judgment"]
classes = sorted(ROTULOS)


def freq_conteudo(d):
    """Contagem de palavras de conteúdo (premissa + hipótese), só palavras alfabéticas."""
    c = contar(pd.concat([d["premise_lemas_sem_sw"], d["hypothesis_lemas_sem_sw"]]))
    return Counter({w: n for w, n in c.items() if e_palavra(w)})


c_geral = freq_conteudo(tr)
c_classe = {c: freq_conteudo(tr[y == c]) for c in classes}

# ----------------------------------------------------------------------
# 1. Palavras mais frequentes
# ----------------------------------------------------------------------
print("=== 1. 20 palavras (lemas sem stopwords) mais frequentes no treino ===")
top = pd.DataFrame(c_geral.most_common(20), columns=["lema", "frequencia"])
print(top.to_string(index=False))
salvar_csv(top, "top_palavras.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.barh(top["lema"][::-1], top["frequencia"][::-1], color="#4C72B0")
ax.set_title("20 lemas mais frequentes (treino)"); ax.set_xlabel("frequência")
fig.tight_layout(); salvar_fig(fig, "fig_top_palavras.png")

fig, eixos = plt.subplots(1, 3, figsize=(14, 4.5))
linhas = []
for ax, c in zip(eixos, classes):
    t = c_classe[c].most_common(12)
    ax.barh([w for w, _ in t][::-1], [n for _, n in t][::-1], color=CORES[c])
    ax.set_title(f"{ROTULOS[c]} (n={int((y == c).sum())})"); ax.set_xlabel("frequência")
    linhas += [(ROTULOS[c], w, n) for w, n in t]
fig.tight_layout(); salvar_fig(fig, "fig_top_palavras_por_classe.png")
salvar_csv(pd.DataFrame(linhas, columns=["classe", "lema", "frequencia"]), "top_palavras_por_classe.csv", index=False)

# ----------------------------------------------------------------------
# 2. Palavras distintivas de cada classe (razão de chances suavizada)
# ----------------------------------------------------------------------
vocab = set(c_geral)
V = len(vocab)
N = {c: sum(c_classe[c].values()) for c in classes}
distintivas = {}
linhas = []
for c in classes:
    resto = Counter()
    for o in classes:
        if o != c:
            resto.update(c_classe[o])
    Nr = sum(resto.values())
    pontos = {}
    for w in vocab:
        if c_geral[w] < 10:  # ignora palavras raras
            continue
        p_c = (c_classe[c][w] + 1) / (N[c] + V)
        p_r = (resto[w] + 1) / (Nr + V)
        pontos[w] = np.log2(p_c / p_r)
    distintivas[c] = pontos
    melhores = sorted(pontos.items(), key=lambda x: -x[1])[:10]
    linhas += [(ROTULOS[c], w, round(s, 2), c_classe[c][w], resto[w]) for w, s in melhores]
dist = pd.DataFrame(linhas, columns=["classe", "lema", "log2_razao", "freq_na_classe", "freq_nas_outras"])
print("\n=== 2. Palavras mais distintivas por classe (log2 da razão de frequências relativas; só palavras com freq >= 10) ===")
for c in classes:
    sub = dist[dist["classe"] == ROTULOS[c]]
    print(f"{ROTULOS[c]:11s}:", ", ".join(f"{r.lema}({r.log2_razao})" for r in sub.itertuples()))
salvar_csv(dist, "palavras_distintivas.csv", index=False)

# ----------------------------------------------------------------------
# 3. Lei de Zipf (tokens)
# ----------------------------------------------------------------------
c_tok = contar(pd.concat([tr["premise_tokens"], tr["hypothesis_tokens"]]))
freqs = np.array(sorted(c_tok.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1)
sel = ranks <= 1000
coef = np.polyfit(np.log10(ranks[sel]), np.log10(freqs[sel]), 1)
print(f"\n=== 3. Zipf ===\nvocabulário={len(freqs)}; inclinação (ranks 1-1000) = {coef[0]:.2f}  (Zipf ideal ≈ -1)")
print(f"palavras que aparecem 1 vez: {(freqs == 1).sum()} ({(freqs == 1).mean() * 100:.1f}% do vocabulário)")
print(f"as 10 palavras mais frequentes cobrem {freqs[:10].sum() / freqs.sum() * 100:.1f}% dos tokens")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(ranks, freqs, ".", markersize=3, color="#4C72B0")
ax.set_xlabel("posição no ranking (log)"); ax.set_ylabel("frequência (log)")
ax.set_title("Frequência x ranking dos tokens (treino)")
fig.tight_layout(); salvar_fig(fig, "fig_zipf.png")

# ----------------------------------------------------------------------
# 4. Negação por classe
# ----------------------------------------------------------------------
def tem_neg(texto):
    return len(NEGACOES & set(texto.split())) > 0

np_ = tr["premise_lemas"].apply(tem_neg)
nh_ = tr["hypothesis_lemas"].apply(tem_neg)
padrao = np.select([np_ & nh_, np_ & ~nh_, ~np_ & nh_], ["nos dois", "só na premissa", "só na hipótese"], "em nenhum")
neg = pd.crosstab(y, padrao, normalize="index").reindex(columns=["em nenhum", "nos dois", "só na premissa", "só na hipótese"]) * 100
neg.index = [ROTULOS[c] for c in neg.index]
print("\n=== 4. Negação por classe (% dos pares) ===")
print(neg.round(1).to_string())
salvar_csv(neg.round(2), "negacao_por_classe.csv")
fig, ax = plt.subplots(figsize=(7, 3.8))
neg.drop(columns="em nenhum").plot(kind="bar", ax=ax, rot=0)
ax.set_ylabel("% dos pares"); ax.set_title("Negação nos pares, por classe (treino)")
fig.tight_layout(); salvar_fig(fig, "fig_negacao_por_classe.png")

# ----------------------------------------------------------------------
# 5. Sobreposição lexical entre premissa e hipótese
# ----------------------------------------------------------------------
def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if (a | b) else 0.0

tr["jaccard"] = [jaccard(a, b) for a, b in zip(tr["premise_lemas_sem_sw"], tr["hypothesis_lemas_sem_sw"])]
sob = tr.groupby(y)["jaccard"].describe().round(3)
sob.index = [ROTULOS[c] for c in sob.index]
print("\n=== 5. Sobreposição de palavras (Jaccard) premissa x hipótese, por classe ===")
print(sob.to_string())
salvar_csv(sob, "sobreposicao_por_classe.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for c in classes:
    a1.hist(tr.loc[y == c, "jaccard"], bins=np.linspace(0, 1, 26), density=True, histtype="step",
            linewidth=1.8, color=CORES[c], label=ROTULOS[c])
a1.set_xlabel("Jaccard"); a1.set_ylabel("densidade"); a1.set_title("Sobreposição lexical"); a1.legend()
a2.boxplot([tr.loc[y == c, "jaccard"] for c in classes])
a2.set_xticklabels([ROTULOS[c] for c in classes]); a2.set_ylabel("Jaccard"); a2.set_title("Por classe")
fig.tight_layout(); salvar_fig(fig, "fig_sobreposicao.png")

# ----------------------------------------------------------------------
# 6. Nuvens de palavras
# ----------------------------------------------------------------------
def nuvem(freq, ax, titulo, cor=None):
    """Nuvem sem palavras funcionais. cor=None usa a paleta viridis; senão, cor fixa."""
    freq = {w: n for w, n in dict(freq).items() if w not in FUNCIONAIS}
    kw = {"colormap": "viridis"} if cor is None else {"color_func": lambda *a, **k: cor}
    wc = WordCloud(width=800, height=500, background_color="white", max_words=80,
                   random_state=42, prefer_horizontal=0.9, **kw)
    wc.generate_from_frequencies(freq)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(titulo)


fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
nuvem(c_geral, eixos[0, 0], "Todas as classes")
for ax, c in zip([eixos[0, 1], eixos[1, 0], eixos[1, 1]], classes):
    nuvem(c_classe[c], ax, ROTULOS[c], CORES[c])
fig.tight_layout(); salvar_fig(fig, "fig_nuvens_frequencia.png")

fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, c in zip(eixos, classes):
    pos = {w: s for w, s in distintivas[c].items() if s > 0}
    nuvem(sorted(pos.items(), key=lambda x: -x[1])[:60], ax, f"{ROTULOS[c]} (palavras distintivas)", CORES[c])
fig.tight_layout(); salvar_fig(fig, "fig_nuvens_distintivas.png")