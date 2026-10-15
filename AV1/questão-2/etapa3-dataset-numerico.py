"""
Etapa 3 - Conversão do corpus em dataset numérico.

Duas representações, ambas ajustadas SOMENTE no treino e aplicadas a validação/teste:
  A) TF-IDF do par (texto da premissa + hipótese, lemas sem stopwords)  -> matriz esparsa
  B) Atributos de RELAÇÃO entre premissa e hipótese (matriz densa, 1 coluna = 1 atributo)

relatedness_score e sentence_pair_id NÃO entram nas matrizes: ficam em meta_*.parquet
(só para análise). O rótulo (y) também fica separado.

Saídas em dados_numericos/: X_tfidf_{split}.npz, X_rel_{split}.parquet, meta_{split}.parquet,
vocabulario_tfidf.csv  |  resultados/atributos_relacao_por_classe.csv
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from eda_utils import (NEGACOES, PASTA_NUM, ROTULOS, SPLITS, carregar, criar_pastas,
                       salvar_csv)

criar_pastas()
dfs = {s: carregar(s) for s in SPLITS}


def texto_par(d):
    return d["premise_lemas_sem_sw"] + " " + d["hypothesis_lemas_sem_sw"]


# ----------------------------------------------------------------------
# A) TF-IDF  (vocabulário e pesos IDF aprendidos só no treino)
# ----------------------------------------------------------------------
vec = TfidfVectorizer(
    token_pattern=r"\S+",   # os textos já estão tokenizados e separados por espaço
    lowercase=False,        # já estão em minúsculas
    min_df=2,               # ignora palavras que aparecem em só 1 par do treino
    sublinear_tf=True,      # 1 + log(tf): atenua repetições
)
vec.fit(texto_par(dfs["train"]))
vocab = np.array(vec.get_feature_names_out())
print("=== A) TF-IDF ===")
print(f"vocabulário (min_df=2, ajustado no treino): {len(vocab)} termos")


# ----------------------------------------------------------------------
# B) Atributos de relação premissa x hipótese
# ----------------------------------------------------------------------
def tem_numero(w):
    return any(ch.isdigit() for ch in w)


def atributos_relacao(d):
    Xp = vec.transform(d["premise_lemas_sem_sw"])
    Xh = vec.transform(d["hypothesis_lemas_sem_sw"])
    cos = np.asarray(Xp.multiply(Xh).sum(axis=1)).ravel()  # linhas já têm norma 1
    linhas = []
    for p, h, tp, th, lp, lh in zip(d["premise_lemas_sem_sw"], d["hypothesis_lemas_sem_sw"],
                                    d["premise_tokens"], d["hypothesis_tokens"],
                                    d["premise_lemas"], d["hypothesis_lemas"]):
        sp, sh = set(p.split()), set(h.split())
        inter = len(sp & sh)
        neg_p = len(NEGACOES & set(lp.split()))
        neg_h = len(NEGACOES & set(lh.split()))
        nums_p = {w for w in tp.split() if tem_numero(w)}
        nums_h = {w for w in th.split() if tem_numero(w)}
        linhas.append({
            "jaccard": inter / len(sp | sh) if (sp | sh) else 0.0,
            "h_contida_em_p": inter / len(sh) if sh else 0.0,
            "p_contida_em_h": inter / len(sp) if sp else 0.0,
            "n_lemas_p": len(sp),
            "n_lemas_h": len(sh),
            "dif_tamanho": len(th.split()) - len(tp.split()),
            "neg_p": neg_p,
            "neg_h": neg_h,
            "negacao_assimetrica": int((neg_p > 0) != (neg_h > 0)),
            "n_numeros_p": len(nums_p),
            "n_numeros_h": len(nums_h),
            "numeros_iguais": int(nums_p == nums_h),
        })
    X = pd.DataFrame(linhas)
    X.insert(0, "cos_tfidf", cos)
    return X


print("\n=== B) Atributos de relação ===")
for split, d in dfs.items():
    X_tfidf = vec.transform(texto_par(d))
    X_rel = atributos_relacao(d)
    meta = d[["sentence_pair_id", "entailment_judgment", "relatedness_score"]].reset_index(drop=True)
    sparse.save_npz(PASTA_NUM / f"X_tfidf_{split}.npz", X_tfidf)
    X_rel.to_parquet(PASTA_NUM / f"X_rel_{split}.parquet", index=False)
    meta.to_parquet(PASTA_NUM / f"meta_{split}.parquet", index=False)
    dens = X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]) * 100
    print(f"{split:10s} TF-IDF {X_tfidf.shape}  não-zeros={dens:.2f}%   |  atributos de relação {X_rel.shape}")
    if split == "train":
        X_rel_tr, meta_tr = X_rel, meta

pd.DataFrame({"termo": vocab}).to_csv(PASTA_NUM / "vocabulario_tfidf.csv", index=False, encoding="utf-8-sig")

print("\nColunas dos atributos de relação:", list(X_rel_tr.columns))
medias = X_rel_tr.groupby(meta_tr["entailment_judgment"]).mean().T
medias.columns = [ROTULOS[c] for c in medias.columns]
print("\nMédia de cada atributo por classe (treino):")
print(medias.round(3).to_string())
salvar_csv(medias.round(4), "atributos_relacao_por_classe.csv")