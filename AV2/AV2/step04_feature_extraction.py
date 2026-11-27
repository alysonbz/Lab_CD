"""
PASSO 4 / TÓPICO 4 - Extração de características e construção do DataFrame.

* Lê as decisões do Passo 2 (pré-processamento) e do Passo 3 (limiar da segmentação);
* extrai, em paralelo e em lotes (retomável), duas variantes por imagem:
    - 'imagem inteira' : cor + textura + gradientes/bordas (443 colunas);
    - 'segmentada'     : as mesmas, restritas à máscara da folha, + forma (457 colunas);
* grava DataFrames em CSV (1 linha por imagem; image_id, class, group_id, split + features).
  O image_id NUNCA entra como entrada do classificador (o Passo 5 só usa colunas de features).

Uso:  python step04_feature_extraction.py [--max-seconds N]
      (--max-seconds interrompe com segurança após N s; basta rodar de novo para continuar)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
import features as F
from utils import log, timer, savefig, save_table, require, split_class


def load_decisions():
    pre = json.loads(config.PREPROC_DECISION_JSON.read_text(encoding="utf-8"))
    seg = json.loads(config.SEG_DECISION_JSON.read_text(encoding="utf-8"))
    return F.PreprocParams(**pre["params"]), float(seg["fator_limiar"])


def feature_descriptions(names_full, names_seg):
    desc = {"rgb_mean": "média do canal RGB (0-1)", "rgb_std": "desvio-padrão do canal RGB",
            "rgb_hist": "histograma RGB (8 bins/canal, normalizado)", "hsv_hue": "matiz médio (média vetorial cos/sin)",
            "hsv_s": "saturação (média/desvio)", "hsv_v": "valor/brilho (média/desvio)",
            "hsv_hist": "histograma de matiz (16) e saturação (8)", "lab_mean": "média em Lab", "lab_std": "desvio em Lab",
            "lbp_P8R1": "histograma LBP uniforme P=8,R=1", "lbp_P16R2": "histograma LBP uniforme P=16,R=2",
            "glcm": "GLCM (32 níveis, 4 ângulos médios): contraste/homogeneidade/energia/correlação/entropia",
            "hog": "HOG (9 bins, células 1/4 da imagem, blocos 2x2)", "edge_density": "densidade de bordas (Canny)",
            "edge_gradmag": "magnitude do gradiente (Sobel): média/desvio", "edge_ori": "histograma de orientação do gradiente (12 bins)",
            "shp_": "forma da máscara: área, perímetro, circularidade, razões de aspecto, solidez, contato com borda, Hu"}
    rows = []
    for n in names_seg:
        d = next((v for k, v in desc.items() if n.startswith(k)), "")
        rows.append({"coluna": n, "grupo": n.split("_")[0], "descricao": d,
                     "na_imagem_inteira": n in set(names_full), "na_segmentada": True})
    return pd.DataFrame(rows)


def extract_all(params, factor, max_seconds=None):
    sp = pd.read_csv(config.SPLIT_CSV)
    paths = [str(config.RAW_COLOR_DIR / i) for i in sp["image_id"]]
    key = f"s{params.size}_{params.denoise}_c{int(params.clahe)}_f{factor}"
    cdir = config.DIR_CACHE / "features" / key
    cdir.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    n_chunks = (n + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
    t0, done_imgs = time.time(), 0
    for k in range(n_chunks):
        f = cdir / f"chunk_{k:04d}.npz"
        if f.exists():
            continue
        if max_seconds and (time.time() - t0) > max_seconds:
            log(f"   limite de tempo ({max_seconds}s) atingido: rode novamente para continuar "
                f"({sum(1 for _ in cdir.glob('chunk_*.npz'))}/{n_chunks} lotes prontos)")
            return None, sp
        a, b = k * config.CHUNK_SIZE, min(n, (k + 1) * config.CHUNK_SIZE)
        Xf, Xs, ok, seg_ok, area = F.extract_parallel(paths[a:b], params, factor, True, True, config.N_JOBS)
        np.savez(f, Xf=Xf, Xs=Xs, ok=ok, seg_ok=seg_ok, area=area)
        done_imgs += b - a
        el = time.time() - t0
        left = (n - b) * el / max(done_imgs, 1)
        log(f"   lote {k + 1}/{n_chunks} ({b}/{n} imagens) | {1000 * el / done_imgs:.1f} ms/img | ETA ≈ {left / 60:.1f} min")
    parts = [np.load(cdir / f"chunk_{k:04d}.npz") for k in range(n_chunks)]
    out = {c: np.concatenate([p[c] for p in parts]) for c in ("Xf", "Xs", "ok", "seg_ok", "area")}
    return out, sp


def write_csv(path, sp, X, names, extra: dict | None = None):
    df = pd.DataFrame(X, columns=names)
    meta = sp[["image_id", "class", "group_id", "split"]].reset_index(drop=True)
    if extra:
        for k, v in extra.items():
            meta[k] = v
    df = pd.concat([meta, df], axis=1)
    df.to_csv(path, index=False, float_format="%.5g")
    log(f"   CSV salvo: {path.name} ({df.shape[0]} linhas x {df.shape[1]} colunas, {path.stat().st_size / 1e6:.0f} MB)")


def plots(sp, Xf, names_f):
    cols = F.columns_for(names_f, config.FEATURE_CONFIGS["C2_COR+TEXTURA"])
    tr = sp.index[sp["split"] == "train"].to_numpy()
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(tr, size=min(6000, len(tr)), replace=False)
    Z = StandardScaler().fit_transform(Xf[idx][:, cols])
    P = PCA(2, random_state=config.SEED).fit(Z)
    E = P.transform(Z)
    crops = sp.loc[idx, "class"].map(lambda c: split_class(c)[0]).to_numpy()
    uc = sorted(set(crops))
    pal = dict(zip(uc, sns.color_palette("tab20", len(uc))))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for c in uc:
        m = crops == c
        ax.scatter(E[m, 0], E[m, 1], s=5, alpha=.5, color=pal[c], label=c)
    ax.set_xlabel(f"PC1 ({100 * P.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * P.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("PCA (somente ilustrativa) de cor+textura, amostra de treino")
    ax.legend(fontsize=6, ncol=2, markerscale=2)
    savefig(fig, "04_pca_ilustrativa.png")
    # médias por classe (z-score) das características de cor/estatísticas mais variáveis
    sel = [i for i in F.columns_for(names_f, ("rgb_mean", "rgb_std", "hsv_hue", "hsv_s", "hsv_v", "lab_", "glcm_"))]
    df = pd.DataFrame(Xf[tr][:, sel], columns=[names_f[i] for i in sel])
    df["class"] = sp.loc[tr, "class"].to_numpy()
    mean = df.groupby("class").mean()
    z = (mean - mean.mean()) / (mean.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(z, cmap="coolwarm", center=0, ax=ax, yticklabels=True, xticklabels=True, cbar_kws={"label": "z-score entre classes"})
    ax.tick_params(labelsize=6)
    ax.set_title("Média por classe (z-score) de características de cor e GLCM — treino")
    savefig(fig, "04_heatmap_medias_por_classe.png")


def main(max_seconds=None):
    params, factor = load_decisions()
    log(f"Parâmetros de pré-processamento: {params.as_dict()} | fator do limiar: {factor}")
    with timer("Extração de características (lotes com retomada)"):
        out, sp = extract_all(params, factor, max_seconds)
    if out is None:
        return False
    S = params.size
    nf, ns = F.feature_names(False, S), F.feature_names(True, S)
    require(out["ok"].all(), f"{(~out['ok']).sum()} imagens não puderam ser lidas.")
    require(np.isfinite(out["Xf"]).all() and np.isfinite(out["Xs"]).all(), "Valores não finitos nas características.")
    log(f"   matriz completa: {out['Xf'].shape} | segmentada: {out['Xs'].shape} | "
        f"máscaras inválidas (fallback): {int((~out['seg_ok']).sum())}")
    with timer("Gravação dos DataFrames (CSV) e matrizes binárias"):
        np.savez(config.FEATURES_FULL_NPZ, X=out["Xf"], names=np.array(nf))
        np.savez(config.FEATURES_SEG_NPZ, X=out["Xs"], names=np.array(ns))
        if not config.FEATURES_FULL_CSV.exists():
            write_csv(config.FEATURES_FULL_CSV, sp, out["Xf"], nf)
        if not config.FEATURES_SEG_CSV.exists():
            write_csv(config.FEATURES_SEG_CSV, sp, out["Xs"], ns,
                      {"seg_ok": out["seg_ok"], "mask_area": out["area"]})
        save_table(feature_descriptions(nf, ns), "04_descricao_features.csv")
        grp = pd.Series([n.split("_")[0] for n in ns]).value_counts().rename_axis("grupo").reset_index(name="n_colunas")
        save_table(grp, "04_colunas_por_grupo.csv")
    with timer("Gráficos de caracterização das características"):
        plots(sp, out["Xf"], nf)
    log("PASSO 4 concluído.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=None)
    main(ap.parse_args().max_seconds)
