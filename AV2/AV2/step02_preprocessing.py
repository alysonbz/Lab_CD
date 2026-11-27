"""
PASSO 2 / TÓPICO 2 - Pré-processamento das imagens.

Cada operação é AVALIADA (não aplicada automaticamente):
  * redimensionamento (64 / 128 / 256 px);
  * redução de ruído (mediana 3x3, gaussiano 3x3);
  * ajuste de contraste (CLAHE no canal L do Lab);
  * conversão de espaços de cor (RGB, HSV, Lab) e escala de cinza (texturas);
  * normalização/padronização das características (com e sem StandardScaler).

A comparação usa SOMENTE imagens de TREINO (subamostra estratificada por grupo) e
validação cruzada agrupada: o conjunto de teste não é tocado.
A decisão é gravada em decisao_preprocessamento.json e lida pelo Passo 4.

Uso:  python step02_preprocessing.py [--force]
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import features as F
from utils import log, timer, savefig, save_table, require

VARIANTS = {
    "P0_128_base":      F.PreprocParams(128, "none", False),
    "P1_64":            F.PreprocParams(64, "none", False),
    "P2_256":           F.PreprocParams(256, "none", False),
    "P3_128_mediana3":  F.PreprocParams(128, "median3", False),
    "P4_128_gauss3":    F.PreprocParams(128, "gauss3", False),
    "P5_128_CLAHE":     F.PreprocParams(128, "none", True),
}
BASELINE = "P0_128_base"
MIN_GAIN = 0.01          # ganho mínimo de F1 macro (absoluto) para abandonar a linha de base


def train_sample(n_target=None):
    sp = pd.read_csv(config.SPLIT_CSV)
    require(len(sp) > 0, "Execute o Passo 1 antes (split não encontrado).")
    tr = sp[sp["split"] == "train"].reset_index(drop=True)
    n_splits = max(2, int(round(len(tr) / (n_target or config.PREPROC_SUBSAMPLE))))
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    _, idx = next(sg.split(np.zeros(len(tr)), tr["class"], tr["group_id"]))
    samp = tr.iloc[idx]
    # garante TODAS as classes (as raras podem não cair na dobra): acrescenta grupos inteiros
    extra = []
    for c in sorted(set(tr["class"]) - set(samp["class"])):
        sub = tr[tr["class"] == c]
        gids = sub["group_id"].drop_duplicates().sample(frac=1.0, random_state=config.SEED)
        take, n = [], 0
        for gid in gids:
            take.append(gid)
            n += int((sub["group_id"] == gid).sum())
            if n >= 60:
                break
        extra.append(sub[sub["group_id"].isin(take)])
    return pd.concat([samp] + extra).reset_index(drop=True)


def cv_scores(X, y, groups, make_model, n_splits=3):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    f1s, accs = [], []
    for tr, te in sg.split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        f1s.append(f1_score(y[te], p, average="macro"))
        accs.append(accuracy_score(y[te], p))
    return np.array(f1s), np.array(accs)


def rf():
    return RandomForestClassifier(n_estimators=150, n_jobs=config.N_JOBS, random_state=config.SEED)


def get_features(name, params, paths, force):
    cache = config.DIR_CACHE / f"prep_{name}.npy"
    if cache.exists() and not force:
        return np.load(cache)
    t = time.time()
    Xf, _, ok, _, _ = F.extract_parallel(paths, params, config.SEED, True, False, config.N_JOBS)
    require(ok.all(), f"{(~ok).sum()} imagens não puderam ser lidas na variante {name}.")
    np.save(cache, Xf)
    log(f"   {name}: {len(paths)} imagens em {time.time() - t:.1f}s ({1000 * (time.time() - t) / len(paths):.1f} ms/img)")
    return Xf


def plot_examples(sample):
    rng = np.random.default_rng(config.SEED)
    pick = sample.groupby("class").head(1).sample(5, random_state=config.SEED)["image_id"].tolist()
    ops = [("original 256", F.PreprocParams(256, "none", False)), ("redim. 128", VARIANTS["P0_128_base"]),
           ("redim. 64", VARIANTS["P1_64"]), ("mediana 3x3", VARIANTS["P3_128_mediana3"]),
           ("gauss 3x3", VARIANTS["P4_128_gauss3"]), ("CLAHE (L)", VARIANTS["P5_128_CLAHE"])]
    fig, axes = plt.subplots(len(pick), len(ops) + 1, figsize=(2.1 * (len(ops) + 1), 2.1 * len(pick)))
    for i, iid in enumerate(pick):
        raw = F.read_image(config.RAW_COLOR_DIR / iid)
        for j, (title, p) in enumerate(ops):
            im = F.preprocess(raw, p)
            axes[i, j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(title, fontsize=8)
        g = cv2.cvtColor(F.preprocess(raw, VARIANTS[BASELINE]), cv2.COLOR_BGR2GRAY)
        axes[i, len(ops)].imshow(g, cmap="gray")
        axes[i, len(ops)].axis("off")
        if i == 0:
            axes[i, len(ops)].set_title("cinza (p/ textura)", fontsize=8)
    savefig(fig, "02_exemplos_preprocessamento.png", dpi=110)
    # efeito do CLAHE no histograma do canal L
    raw = F.read_image(config.RAW_COLOR_DIR / pick[0])
    a = cv2.cvtColor(F.preprocess(raw, VARIANTS[BASELINE]), cv2.COLOR_BGR2LAB)[..., 0]
    b = cv2.cvtColor(F.preprocess(raw, VARIANTS["P5_128_CLAHE"]), cv2.COLOR_BGR2LAB)[..., 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(a.ravel(), 64, alpha=.6, label="sem CLAHE")
    ax.hist(b.ravel(), 64, alpha=.6, label="com CLAHE")
    ax.set_title("Canal L: histograma antes/depois do CLAHE")
    ax.legend()
    savefig(fig, "02_histograma_clahe.png")


def main(force=False):
    sample = train_sample()
    log(f"Subamostra de TREINO para decisões de pré-processamento: {len(sample)} imagens, "
        f"{sample['class'].nunique()} classes, {sample['group_id'].nunique()} grupos")
    paths = [str(config.RAW_COLOR_DIR / i) for i in sample["image_id"]]
    y, groups = sample["class"].to_numpy(), sample["group_id"].to_numpy()
    names = F.feature_names(False)
    c1 = F.columns_for(names, config.FEATURE_CONFIGS["C1_COR"])
    c2 = F.columns_for(names, config.FEATURE_CONFIGS["C2_COR+TEXTURA"])

    with timer("Exemplos visuais antes/depois"):
        plot_examples(sample)

    rows, feats = [], {}
    with timer("Comparação de variantes de pré-processamento (RF, CV agrupada 3 dobras)"):
        for name, p in VARIANTS.items():
            X = get_features(name, p, paths, force)
            feats[name] = X
            for cfg, cols in (("C1_COR", c1), ("C2_COR+TEXTURA", c2)):
                f1, acc = cv_scores(X[:, cols], y, groups, rf)
                rows.append({"variante": name, "tamanho": p.size, "denoise": p.denoise, "clahe": p.clahe,
                             "config": cfg, "f1_macro_media": f1.mean(), "f1_macro_dp": f1.std(),
                             "acuracia_media": acc.mean()})
                log(f"   {name:18s} {cfg:16s} F1macro={f1.mean():.4f}±{f1.std():.4f} acc={acc.mean():.4f}")
        res = pd.DataFrame(rows)
        save_table(res, "02_comparacao_preprocessamento.csv")

    # decisão (parcimônia): só abandona a linha de base se o ganho em C2 for >= MIN_GAIN
    c2r = res[res["config"] == "C2_COR+TEXTURA"].set_index("variante")["f1_macro_media"]
    best = c2r.idxmax()
    gain = float(c2r[best] - c2r[BASELINE])
    chosen = best if (best != BASELINE and gain >= MIN_GAIN) else BASELINE
    log(f"   melhor variante={best} (ganho vs base={gain:+.4f}) -> escolhida: {chosen}")

    fig, ax = plt.subplots(figsize=(9, 4))
    piv = res.pivot(index="variante", columns="config", values="f1_macro_media").loc[list(VARIANTS)]
    piv.plot.bar(ax=ax)
    ax.set_ylabel("F1 macro (CV agrupada, treino)")
    ax.set_title("Efeito do pré-processamento no desempenho")
    ax.set_ylim(max(0, piv.min().min() - 0.05), piv.max().max() + 0.03)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(fig, "02_comparacao_preprocessamento.png")

    with timer("Espaços de cor (variante base)"):
        X = feats[BASELINE]
        sets = {"RGB": ("rgb_",), "HSV": ("hsv_",), "Lab": ("lab_",),
                "RGB+HSV+Lab": ("rgb_", "hsv_", "lab_"), "RGB+HSV+Lab+LBP+GLCM": config.FEATURE_CONFIGS["C2_COR+TEXTURA"]}
        r2 = []
        for k, pref in sets.items():
            cols = F.columns_for(names, pref)
            f1, acc = cv_scores(X[:, cols], y, groups, rf)
            r2.append({"conjunto": k, "n_features": len(cols), "f1_macro_media": f1.mean(), "f1_macro_dp": f1.std()})
            log(f"   {k:24s} n={len(cols):3d} F1macro={f1.mean():.4f}")
        save_table(pd.DataFrame(r2), "02_comparacao_espacos_cor.csv")

    with timer("Normalização (padronização) por algoritmo"):
        r3 = []
        for mname, mk_raw, mk_std in (
                ("KNN(k=7)", lambda: KNeighborsClassifier(7), lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(7))),
                ("SVM-RBF(C=10)", lambda: SVC(C=10), lambda: make_pipeline(StandardScaler(), SVC(C=10))),
                ("RandomForest", rf, lambda: make_pipeline(StandardScaler(), rf()))):
            for tag, mk in (("sem padronização", mk_raw), ("com StandardScaler", mk_std)):
                f1, acc = cv_scores(X[:, c2], y, groups, mk)
                r3.append({"modelo": mname, "tratamento": tag, "f1_macro_media": f1.mean(), "f1_macro_dp": f1.std()})
                log(f"   {mname:14s} {tag:20s} F1macro={f1.mean():.4f}")
        save_table(pd.DataFrame(r3), "02_efeito_padronizacao.csv")

    decision = {"escolhida": chosen, "params": VARIANTS[chosen].as_dict(), "melhor_variante_cv": best,
                "ganho_melhor_vs_base_f1macro": gain, "regra": f"abandona a base somente se ganho >= {MIN_GAIN}",
                "n_imagens_subamostra": int(len(sample))}
    config.PREPROC_DECISION_JSON.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"PASSO 2 concluído. Decisão: {decision['params']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
