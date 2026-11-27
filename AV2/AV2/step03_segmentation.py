"""
PASSO 3 / TÓPICO 3 - Segmentação (somente métodos tradicionais).

Método final: limiarização de Otsu sobre o CROMA do espaço Lab (fundo neutro x folha
colorida), com fator de calibração do limiar, abertura/fechamento morfológico, remoção
de componentes pequenos e preenchimento de buracos por contornos.

Avaliação:
  (a) qualitativa  - original / máscara / resultado / contorno, melhores e piores casos;
  (b) quantitativa - IoU e Dice contra as máscaras que acompanham o repositório
      (raw/segmented). ATENÇÃO: são máscaras geradas pelos autores do dataset (um
      PROXY, não anotação manual) e são usadas SOMENTE para calibrar/avaliar o limiar;
      nunca entram como característica;
  (c) impacto na classificação - CV agrupada, apenas treino: imagem inteira x segmentada.

Uso:  python step03_segmentation.py
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

import config
import features as F
from step02_preprocessing import cv_scores, rf
from utils import log, timer, savefig, save_table, require, stem_no_ext

RAW_BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/segmented/"


# ------------------------- métodos candidatos -------------------------------- #
def seg_hsv_sat(img):
    """Otsu na saturação (HSV)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    _, m = cv2.threshold(hsv[..., 1], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return F._clean_mask(m).astype(bool)


def seg_kmeans_lab(img):
    """Agrupamento de pixels: k-means (k=2) em Lab; folha = cluster de maior croma."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    X = lab.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(config.SEED)
    _, lbl, cen = cv2.kmeans(X, 2, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(img.shape[:2])
    k = int(np.argmax([np.hypot(c[1] - 128, c[2] - 128) for c in cen]))
    return F._clean_mask((lbl == k).astype(np.uint8)).astype(bool)


def seg_lab_chroma(img, factor):
    return F.segment_leaf(img, factor)[0]


# ------------------------- referência (proxy) -------------------------------- #
def _ref_path(image_id):
    cls, fn = image_id.split("/", 1)
    return config.REPO_DIR / "raw" / "segmented" / cls / (stem_no_ext(fn) + "_final_masked.jpg")


def _get_ref(image_id):
    p = _ref_path(image_id)
    if not p.exists():
        cls, fn = image_id.split("/", 1)
        url = RAW_BASE + urllib.parse.quote(cls) + "/" + urllib.parse.quote(stem_no_ext(fn) + "_final_masked.jpg")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=20) as r:
                p.write_bytes(r.read())
        except Exception:
            return None
    r = F.read_image(p)
    return None if r is None else r


def load_pairs(ids, size, want):
    """Devolve lista de (image_id, img_bgr, ref_mask_bool) com referência disponível."""
    with ThreadPoolExecutor(8) as ex:
        refs = list(ex.map(_get_ref, ids))
    out = []
    for iid, r in zip(ids, refs):
        if r is None:
            continue
        r = cv2.resize(r, (size, size), interpolation=cv2.INTER_AREA)
        img = F.preprocess(F.read_image(config.RAW_COLOR_DIR / iid), F.PreprocParams(size))
        out.append((iid, img, r.max(2) > 12))
        if len(out) >= want:
            break
    return out


def iou_dice(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return (inter / uni if uni else 1.0), (2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) else 1.0)


def overlay_contour(img, mask):
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def main():
    pre = json.loads(config.PREPROC_DECISION_JSON.read_text(encoding="utf-8"))
    params = F.PreprocParams(**pre["params"])
    S = params.size
    sp = pd.read_csv(config.SPLIT_CSV)
    tr = sp[sp["split"] == "train"].reset_index(drop=True)
    rng = np.random.default_rng(config.SEED)
    # candidatos: imagens de TREINO, embaralhadas (3x o necessário p/ compensar referências ausentes)
    cand = tr.sample(frac=1.0, random_state=config.SEED)["image_id"].tolist()[:config.SEG_REF_SAMPLE * 4]

    with timer("Referências (máscaras do repositório, proxy) para calibração/avaliação"):
        pairs = load_pairs(cand, S, config.SEG_REF_SAMPLE * 2)
        log(f"   {len(pairs)} pares imagem/referência disponíveis")
    have_ref = len(pairs) >= 20
    factor = config.SEG_DEFAULT_FACTOR
    results = {}
    if have_ref:
        half = len(pairs) // 2
        calib, valid = pairs[:half], pairs[half:]
        with timer("Calibração do fator do limiar (metade A) e avaliação (metade B)"):
            rows = []
            for f in config.SEG_FACTOR_GRID:
                v = [iou_dice(seg_lab_chroma(im, f), ref)[0] for _, im, ref in calib]
                rows.append({"fator": f, "iou_medio_calibracao": float(np.mean(v))})
                log(f"   fator={f:.2f}  IoU(calibração)={np.mean(v):.4f}")
            cal = pd.DataFrame(rows)
            save_table(cal, "03_calibracao_fator_limiar.csv")
            factor = float(cal.loc[cal["iou_medio_calibracao"].idxmax(), "fator"])
            log(f"   fator escolhido = {factor}")
            methods = {"Otsu em S (HSV)": seg_hsv_sat, "k-means (k=2) em Lab": seg_kmeans_lab,
                       "Otsu no croma Lab (f=1.0)": lambda im: seg_lab_chroma(im, 1.0),
                       f"Otsu no croma Lab (f={factor}) [adotado]": lambda im: seg_lab_chroma(im, factor)}
            rows, per_img = [], {}
            for name, fn in methods.items():
                vals = [iou_dice(fn(im), ref) for _, im, ref in valid]
                ious = np.array([v[0] for v in vals])
                dices = np.array([v[1] for v in vals])
                per_img[name] = ious
                rows.append({"metodo": name, "n_imagens": len(valid), "iou_medio": ious.mean(),
                             "iou_mediana": np.median(ious), "iou_p10": np.percentile(ious, 10),
                             "dice_medio": dices.mean(), "frac_iou_abaixo_0.7": float((ious < 0.7).mean())})
            comp = pd.DataFrame(rows)
            save_table(comp, "03_comparacao_metodos_segmentacao.csv")
            log("\n" + comp.round(3).to_string(index=False))
            results = comp.iloc[-1].to_dict()
            adopted = list(methods)[-1]
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.hist(per_img[adopted], bins=25, color="#4c72b0")
            ax.set_xlabel("IoU com a máscara de referência")
            ax.set_ylabel("imagens")
            ax.set_title("Distribuição do IoU (método adotado, metade B)")
            savefig(fig, "03_distribuicao_iou.png")
        # melhores e piores casos
        ious = per_img[adopted]
        order = np.argsort(ious)
        show = list(order[-4:][::-1]) + list(order[:4])
        fig, axes = plt.subplots(4, len(show), figsize=(2.0 * len(show), 8.4))
        for j, k in enumerate(show):
            iid, im, ref = valid[k]
            m = seg_lab_chroma(im, factor)
            res = im * m[..., None]
            axes[0, j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); axes[0, j].set_title(
                ("melhor" if j < 4 else "pior") + f"\nIoU={ious[k]:.2f}", fontsize=8)
            axes[1, j].imshow(m, cmap="gray")
            axes[2, j].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            axes[3, j].imshow(ref, cmap="gray")
            for i in range(4):
                axes[i, j].axis("off")
        for i, t in enumerate(["original", "máscara (nossa)", "resultado", "referência (proxy)"]):
            axes[i, 0].text(-0.08, 0.5, t, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
        savefig(fig, "03_melhores_e_piores_casos.png", dpi=110)

    # ---- exemplos gerais (uma imagem de classes variadas), sem referência necessária ----
    with timer("Exemplos de segmentação (original, máscara, resultado, contorno)"):
        pick = tr.groupby("class").head(1).sample(8, random_state=config.SEED)["image_id"].tolist()
        fig, axes = plt.subplots(4, len(pick), figsize=(2.0 * len(pick), 8.2))
        for j, iid in enumerate(pick):
            im = F.preprocess(F.read_image(config.RAW_COLOR_DIR / iid), params)
            m, ok = F.segment_leaf(im, factor)
            axes[0, j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); axes[0, j].set_title(iid.split("/")[0][:16], fontsize=6)
            axes[1, j].imshow(m, cmap="gray")
            axes[2, j].imshow(cv2.cvtColor(im * m[..., None], cv2.COLOR_BGR2RGB))
            axes[3, j].imshow(overlay_contour(im, m))
            for i in range(4):
                axes[i, j].axis("off")
        for i, t in enumerate(["original", "máscara", "resultado", "contorno"]):
            axes[i, 0].text(-0.08, 0.5, t, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
        savefig(fig, "03_exemplos_segmentacao.png", dpi=110)

    # ---- limitações: estatísticas em uma amostra maior de treino ----
    with timer("Limitações: área da máscara e falhas (amostra de treino)"):
        smp = tr.sample(n=min(1500, len(tr)), random_state=config.SEED)
        _, Xs, ok, seg_ok, area = F.extract_parallel([str(config.RAW_COLOR_DIR / i) for i in smp["image_id"]],
                                                     params, factor, False, True, config.N_JOBS)
        lim = {"n": int(len(smp)), "frac_fallback_mascara_invalida": float((~seg_ok).mean()),
               "frac_folha_ocupa_mais_de_90pct": float((area > 0.90).mean()),
               "frac_folha_menos_de_15pct": float((area < 0.15).mean()),
               "area_mediana": float(np.nanmedian(area))}
        log(f"   {lim}")
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.hist(area[~np.isnan(area)], bins=30, color="#55a868")
        ax.set_xlabel("fração da imagem ocupada pela máscara")
        ax.set_ylabel("imagens")
        ax.set_title("Área da folha segmentada (amostra de treino)")
        savefig(fig, "03_area_mascara.png")

    # ---- impacto na classificação (CV agrupada no TREINO) ----
    with timer("Impacto da segmentação na classificação (CV agrupada, apenas treino)"):
        from step02_preprocessing import train_sample
        sample = train_sample()
        paths = [str(config.RAW_COLOR_DIR / i) for i in sample["image_id"]]
        Xf, Xs, ok, _, _ = F.extract_parallel(paths, params, factor, True, True, config.N_JOBS)
        require(ok.all(), "Falha de leitura na subamostra.")
        y, g = sample["class"].to_numpy(), sample["group_id"].to_numpy()
        nf, ns = F.feature_names(False, S), F.feature_names(True, S)
        rows = []
        for cfg, pref in config.FEATURE_CONFIGS.items():
            for var, X, names in (("imagem inteira", Xf, nf), ("segmentada", Xs, ns)):
                cols = F.columns_for(names, pref)
                if len(cols) == 0 or (var == "imagem inteira" and cfg.startswith("C4")):
                    continue
                f1, acc = cv_scores(X[:, cols], y, g, rf)
                rows.append({"config": cfg, "variante": var, "n_features": len(cols),
                             "f1_macro_media": f1.mean(), "f1_macro_dp": f1.std(), "acuracia_media": acc.mean()})
                log(f"   {cfg:28s} {var:15s} F1macro={f1.mean():.4f}±{f1.std():.4f}")
        seg_cv = pd.DataFrame(rows)
        save_table(seg_cv, "03_segmentacao_vs_inteira_cv.csv")

    dec = {"fator_limiar": factor, "usou_referencia": bool(have_ref), "metricas_metodo_adotado": results,
           "limitacoes": lim, "params_preprocessamento": params.as_dict()}
    config.SEG_DECISION_JSON.write_text(json.dumps(dec, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"PASSO 3 concluído. Fator do limiar = {factor}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
