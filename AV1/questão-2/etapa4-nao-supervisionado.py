"""
Etapa 4 - Aplicações de aprendizado NÃO supervisionado sobre o dataset numérico (treino).

  Aplicação 1 - Redução de dimensionalidade e visualização (SVD, PCA, t-SNE)
  Aplicação 2 - Agrupamento (K-Means): estrutura de relação e temas de conteúdo
  Aplicação 3 - Modelagem de tópicos (NMF)
  Aplicação 4 - Similaridade entre frases (cosseno TF-IDF) como medida não supervisionada
  Aplicação 5 - Detecção de anomalias (Isolation Forest): pares atípicos / possíveis erros de anotação
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, homogeneity_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)
from sklearn.preprocessing import Normalizer, StandardScaler

from eda_utils import (CORES, PASTA_NUM, ROTULOS, SEMENTE, carregar, criar_pastas,
                       exigir_arquivo, salvar_csv, salvar_fig)

criar_pastas()
dica = "Rode antes a etapa3-dataset-numerico.py."
X_tfidf = sparse.load_npz(exigir_arquivo(PASTA_NUM / "X_tfidf_train.npz", dica))
X_rel = pd.read_parquet(exigir_arquivo(PASTA_NUM / "X_rel_train.parquet", dica))
meta = pd.read_parquet(exigir_arquivo(PASTA_NUM / "meta_train.parquet", dica))
vocab = pd.read_csv(PASTA_NUM / "vocabulario_tfidf.csv", encoding="utf-8-sig")["termo"].to_numpy()
y = meta["entailment_judgment"].to_numpy()
classes = sorted(ROTULOS)
tr = carregar("train")  # só para mostrar frases na aplicação 5

Z_rel = StandardScaler().fit_transform(X_rel)  # atributos de relação padronizados (média 0, desvio 1)


def dispersao(ax, P, titulo):
    for c in classes:
        m = y == c
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.5 if c == 0 else 0.8, color=CORES[c],
                   label=ROTULOS[c], zorder=3 if c else 2)
    ax.set_title(titulo); ax.set_xticks([]); ax.set_yticks([])


# ======================================================================
print("=" * 70, "\nAPLICAÇÃO 1 - Redução de dimensionalidade e visualização\n" + "=" * 70)
svd = TruncatedSVD(n_components=100, random_state=SEMENTE).fit(X_tfidf)
acum = np.cumsum(svd.explained_variance_ratio_)
print("TF-IDF -> SVD: variância explicada acumulada:",
      {k: f"{acum[k - 1] * 100:.1f}%" for k in (2, 10, 50, 100)})
P_svd = svd.transform(X_tfidf)[:, :2]

pca = PCA(n_components=len(X_rel.columns), random_state=SEMENTE).fit(Z_rel)
print("Atributos de relação -> PCA: variância explicada por componente:",
      [f"{v * 100:.1f}%" for v in pca.explained_variance_ratio_[:4]], "(PC1..PC4)")
carga = pd.DataFrame(pca.components_[:2].T, index=X_rel.columns, columns=["PC1", "PC2"])
print("Atributos que mais pesam em cada componente:")
for pc in ("PC1", "PC2"):
    top = carga[pc].abs().sort_values(ascending=False).head(4).index
    print(f"  {pc}:", ", ".join(f"{a}({carga.loc[a, pc]:+.2f})" for a in top))
P_pca = pca.transform(Z_rel)[:, :2]
P_tsne = TSNE(n_components=2, perplexity=30, random_state=SEMENTE, init="pca").fit_transform(Z_rel)

# separação por classe em cada representação (silhouette usando o RÓTULO só como avaliação)
for nome, M in [("TF-IDF (SVD 2D)", P_svd), ("relação (PCA 2D)", P_pca), ("relação (t-SNE 2D)", P_tsne)]:
    print(f"  silhouette dos rótulos em {nome}: {silhouette_score(M, y):+.3f}")
print("  (perto de 0 = classes sobrepostas; negativo = misturadas)")

fig, eixos = plt.subplots(1, 3, figsize=(16, 4.6))
dispersao(eixos[0], P_svd, "TF-IDF do par -> SVD (2 comp.)")
dispersao(eixos[1], P_pca, "Atributos de relação -> PCA")
dispersao(eixos[2], P_tsne, "Atributos de relação -> t-SNE")
eixos[0].legend(markerscale=2)
fig.tight_layout(); salvar_fig(fig, "fig_reducao_dimensionalidade.png")

# ======================================================================
print("\n" + "=" * 70, "\nAPLICAÇÃO 2 - Agrupamento (K-Means)\n" + "=" * 70)
def melhor_k(M, ks):
    notas = {}
    for k in ks:
        rot = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit_predict(M)
        notas[k] = silhouette_score(M, rot)
    return notas

# (a) estrutura de RELAÇÃO
notas_rel = melhor_k(Z_rel, range(2, 9))
k_rel = max(notas_rel, key=notas_rel.get)
print("(a) Atributos de relação - silhouette por k:", {k: round(v, 3) for k, v in notas_rel.items()})
print(f"    melhor k = {k_rel}")
resultados = []
for k in sorted({3, k_rel}):
    rot = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit_predict(Z_rel)
    resultados.append((k, rot))
    print(f"    k={k}: ARI={adjusted_rand_score(y, rot):.3f}  NMI={normalized_mutual_info_score(y, rot):.3f}"
          f"  homogeneidade={homogeneity_score(y, rot):.3f}")
k_a, rot_rel = resultados[0]  # k=3 (3 classes) para a tabela de contingência
tab = pd.crosstab(rot_rel, y, normalize="index") * 100
tab.columns = [ROTULOS[c] for c in tab.columns]
tab.index.name = "cluster"
tab.insert(0, "n", pd.Series(rot_rel).value_counts().sort_index())
print("    Composição dos clusters (k=3), % de cada classe dentro do cluster:")
print(tab.round(1).to_string())
salvar_csv(tab.round(2), "kmeans_relacao_composicao.csv")
centros = X_rel.groupby(rot_rel).mean()  # perfil de cada cluster nos valores originais
print("    Perfil médio (valores originais) de cada cluster:")
print(centros[["cos_tfidf", "jaccard", "h_contida_em_p", "dif_tamanho", "negacao_assimetrica", "numeros_iguais"]].round(2).to_string())

# (b) TEMAS de conteúdo
M_cont = Normalizer().fit_transform(TruncatedSVD(n_components=100, random_state=SEMENTE).fit_transform(X_tfidf))
notas_cont = melhor_k(M_cont, range(4, 13))
k_cont = max(notas_cont, key=notas_cont.get)
print("\n(b) Conteúdo (TF-IDF->SVD) - silhouette por k:", {k: round(v, 3) for k, v in notas_cont.items()})
rot_cont = KMeans(n_clusters=k_cont, n_init=10, random_state=SEMENTE).fit_predict(M_cont)
print(f"    melhor k = {k_cont}; ARI vs. classes={adjusted_rand_score(y, rot_cont):.3f}  "
      f"NMI={normalized_mutual_info_score(y, rot_cont):.3f}")
Xr = X_tfidf.tocsr()
linhas = []
for c in range(k_cont):
    idx = np.where(rot_cont == c)[0]
    med = np.asarray(Xr[idx].mean(axis=0)).ravel()
    termos = ", ".join(vocab[np.argsort(-med)[:8]])
    linhas.append((c, len(idx), termos))
    print(f"    cluster {c} (n={len(idx)}): {termos}")
salvar_csv(pd.DataFrame(linhas, columns=["cluster", "n", "termos_principais"]), "kmeans_conteudo_termos.csv", index=False)

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.2))
a1.plot(list(notas_rel), list(notas_rel.values()), "o-", label="relação")
a1.plot(list(notas_cont), list(notas_cont.values()), "s-", label="conteúdo")
a1.set_xlabel("k"); a1.set_ylabel("silhouette"); a1.set_title("Escolha de k"); a1.legend()
im = a2.imshow(tab.drop(columns="n").to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
a2.set_xticks(range(3)); a2.set_xticklabels([ROTULOS[c] for c in classes])
a2.set_yticks(range(len(tab))); a2.set_yticklabels([f"cluster {i} (n={n})" for i, n in zip(tab.index, tab["n"])])
for i in range(tab.shape[0]):
    for j in range(3):
        a2.text(j, i, f"{tab.iloc[i, j + 1]:.0f}%", ha="center", va="center", fontsize=10)
a2.set_title("K-Means (relação, k=3): classes por cluster")
cores_k = plt.cm.tab10(np.arange(3))
for k in range(3):
    m = rot_rel == k
    a3.scatter(P_pca[m, 0], P_pca[m, 1], s=7, alpha=0.6, color=cores_k[k], label=f"cluster {k}")
a3.set_title("Clusters de relação no plano PCA"); a3.legend(markerscale=2); a3.set_xticks([]); a3.set_yticks([])
fig.tight_layout(); salvar_fig(fig, "fig_kmeans.png")

# ======================================================================
print("\n" + "=" * 70, "\nAPLICAÇÃO 3 - Modelagem de tópicos (NMF)\n" + "=" * 70)
K_TOP = 8
nmf = NMF(n_components=K_TOP, init="nndsvd", random_state=SEMENTE, max_iter=500)
W = nmf.fit_transform(X_tfidf)
Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)  # peso relativo de cada tópico por par
linhas = []
for t in range(K_TOP):
    termos = ", ".join(vocab[np.argsort(-nmf.components_[t])[:10]])
    linhas.append((t, termos))
    print(f"  tópico {t}: {termos}")
salvar_csv(pd.DataFrame(linhas, columns=["topico", "termos"]), "topicos_nmf.csv", index=False)
por_classe = pd.DataFrame({ROTULOS[c]: Wn[y == c].mean(axis=0) for c in classes}) * 100
por_classe.index = [f"tópico {t}" for t in range(K_TOP)]
print("\n  Peso médio de cada tópico por classe (%):")
print(por_classe.round(1).to_string())
salvar_csv(por_classe.round(2), "topicos_por_classe.csv")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(por_classe.to_numpy(), cmap="Blues", aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(por_classe.columns); ax.set_yticks(range(K_TOP)); ax.set_yticklabels(por_classe.index)
for i in range(K_TOP):
    for j in range(3):
        ax.text(j, i, f"{por_classe.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9)
ax.set_title("Peso médio dos tópicos NMF por classe (%)"); fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout(); salvar_fig(fig, "fig_topicos_por_classe.png")

# ======================================================================
print("\n" + "=" * 70, "\nAPLICAÇÃO 4 - Similaridade não supervisionada (cosseno TF-IDF)\n" + "=" * 70)
cos = X_rel["cos_tfidf"].to_numpy()
rel = meta["relatedness_score"].to_numpy()
rho, _ = spearmanr(cos, rel)
print(f"  Spearman(cosseno, relatedness_score) = {rho:.3f}")
rho_j, _ = spearmanr(X_rel["jaccard"], rel)
print(f"  Spearman(Jaccard, relatedness_score) = {rho_j:.3f}")
print(f"  AUC do cosseno para PARAPHRASE x resto = {roc_auc_score(y == 2, cos):.3f}")
print(f"  AUC do cosseno para ENTAILMENT+PARAPHRASE x NONE = {roc_auc_score(y > 0, cos):.3f}")
print("  Cosseno médio por classe:", {ROTULOS[c]: round(float(cos[y == c].mean()), 3) for c in classes})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
rng = np.random.default_rng(SEMENTE)
for c in classes:
    m = y == c
    a1.scatter(cos[m], rel[m] + rng.normal(0, 0.05, m.sum()), s=7, alpha=0.5, color=CORES[c], label=ROTULOS[c])
a1.set_xlabel("cosseno TF-IDF (premissa, hipótese)"); a1.set_ylabel("relatedness_score (com ruído para visualização)")
a1.set_title(f"Similaridade lexical x similaridade humana (ρ={rho:.2f})"); a1.legend(markerscale=2)
a2.boxplot([cos[y == c] for c in classes]); a2.set_xticklabels([ROTULOS[c] for c in classes])
a2.set_ylabel("cosseno TF-IDF"); a2.set_title("Cosseno por classe")
fig.tight_layout(); salvar_fig(fig, "fig_similaridade.png")

# ======================================================================
print("\n" + "=" * 70, "\nAPLICAÇÃO 5 - Detecção de anomalias (Isolation Forest)\n" + "=" * 70)
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=SEMENTE).fit(Z_rel)
escore = -iso.score_samples(Z_rel)  # maior = mais anômalo
flag = iso.predict(Z_rel) == -1
print(f"  pares sinalizados como atípicos: {flag.sum()} ({flag.mean() * 100:.1f}%)")
print("  distribuição por classe dos sinalizados:",
      {ROTULOS[c]: int((flag & (y == c)).sum()) for c in classes})
print("  taxa de sinalização por classe (%):",
      {ROTULOS[c]: round(float(flag[y == c].mean() * 100), 1) for c in classes})
print("\n  5 pares mais atípicos:")
linhas = []
for i in np.argsort(-escore)[:5]:
    r = tr.iloc[i]
    print(f"   [{i}] classe={ROTULOS[y[i]]} relatedness={rel[i]}")
    print(f"       P: {r['premise']}\n       H: {r['hypothesis']}")
for i in np.argsort(-escore)[:20]:
    r = tr.iloc[i]
    linhas.append((i, ROTULOS[y[i]], rel[i], round(float(escore[i]), 3), r["premise"], r["hypothesis"]))
salvar_csv(pd.DataFrame(linhas, columns=["indice", "classe", "relatedness", "escore_anomalia", "premissa", "hipotese"]),
           "pares_atipicos.csv", index=False)