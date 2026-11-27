"""
features.py - Pré-processamento, segmentação tradicional e extração de características.

Somente técnicas clássicas de PDI (OpenCV / NumPy / scikit-image). Sem CNN,
sem Deep Learning, sem modelos pré-treinados.

Convenções
----------
* Imagens em BGR uint8 (convenção do OpenCV).
* "mask" é um array booleano (H, W): True = folha.
* Cada característica pertence a um GRUPO identificado pelo prefixo do nome:
  rgb_, hsv_, lab_, lbp_, glcm_, hog_, edge_, shp_.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

cv2.setNumThreads(1)          # evita sobre-assinatura quando há vários processos

# Funções do OpenCV usadas neste projeto (HOGDescriptor é opcional: há fallback em NumPy).
_REQUIRED_CV2 = ["imdecode", "resize", "medianBlur", "GaussianBlur", "createCLAHE", "cvtColor", "threshold",
                 "morphologyEx", "getStructuringElement", "connectedComponentsWithStats", "findContours",
                 "drawContours", "moments", "HuMoments", "minAreaRect", "convexHull", "arcLength", "contourArea",
                 "Sobel", "cartToPolar", "Canny", "erode", "kmeans", "setRNGSeed", "INTER_AREA", "COLOR_BGR2LAB",
                 "THRESH_OTSU", "MORPH_ELLIPSE", "KMEANS_PP_CENTERS"]
_missing = [n for n in _REQUIRED_CV2 if not hasattr(cv2, n)]
if _missing:
    raise ImportError(f"OpenCV {cv2.__version__} não possui: {_missing}. Instale uma versão compatível: "
                      "pip install \"opencv-python>=4.7,<5\"")

# =========================================================================== #
# 1) PRÉ-PROCESSAMENTO
# =========================================================================== #


@dataclass(frozen=True)
class PreprocParams:
    size: int = 128              # lado da imagem quadrada
    denoise: str = "none"        # "none" | "median3" | "gauss3"
    clahe: bool = False          # CLAHE no canal L (Lab)

    def as_dict(self):
        return asdict(self)


def read_image(path) -> np.ndarray | None:
    """Lê imagem BGR com np.fromfile (robusto a caminhos com acentos/espaços)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def preprocess(img: np.ndarray, p: PreprocParams) -> np.ndarray:
    """Aplica redimensionamento (+ denoise e CLAHE opcionais)."""
    if img.shape[0] != p.size or img.shape[1] != p.size:
        # INTER_AREA é o interpolador recomendado para reduzir (evita aliasing).
        img = cv2.resize(img, (p.size, p.size), interpolation=cv2.INTER_AREA)
    if p.denoise == "median3":
        img = cv2.medianBlur(img, 3)
    elif p.denoise == "gauss3":
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if p.clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = cl.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


# =========================================================================== #
# 2) SEGMENTAÇÃO TRADICIONAL (limiarização de Otsu no croma Lab + morfologia)
# =========================================================================== #


def _lab_chroma_u8(img: np.ndarray) -> np.ndarray:
    """Croma em Lab, sqrt((a-128)^2+(b-128)^2), normalizado para 0..255.

    O fundo do PlantVillage é uma superfície quase neutra (cinza/arroxeada),
    de croma baixo; a folha (verde, amarela, marrom) tem croma maior.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ch = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    mx = float(ch.max())
    if mx < 1e-6:
        return np.zeros(ch.shape, np.uint8)
    return np.clip(ch * (255.0 / mx), 0, 255).astype(np.uint8)


def _clean_mask(m: np.ndarray) -> np.ndarray:
    """Abertura+fechamento, remove componentes pequenos e preenche buracos (contornos)."""
    s = m.shape[0]
    k = max(3, (s // 26) | 1)                       # 128 -> 5 ; 64 -> 3 ; 256 -> 9
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        keep = np.zeros(n, bool)
        keep[1:] = (areas >= 0.15 * big) & (areas >= 0.02 * m.size)
        m = keep[lab].astype(np.uint8)
    # preenche buracos internos usando contornos externos
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out


def segment_leaf(img: np.ndarray, factor: float = 0.8):
    """Retorna (mask_bool, ok). `factor` multiplica o limiar de Otsu (calibrado em treino).

    ok=False indica máscara não confiável (muito pequena); nesse caso a máscara
    devolvida é a imagem inteira (fallback seguro) e o chamador registra o fato.
    """
    c8 = _lab_chroma_u8(img)
    t, _ = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (c8 > t * factor).astype(np.uint8)
    m = _clean_mask(m)
    if m.mean() < 0.02:
        return np.ones(m.shape, bool), False
    return m.astype(bool), True


# =========================================================================== #
# 3) CARACTERÍSTICAS
# =========================================================================== #

_RGB_BINS, _H_BINS, _S_BINS = 8, 16, 8
_LBP_CFG = ((8, 1), (16, 2))                      # (P, R) -> uniform
_GLCM_LEVELS = 32
_GLCM_DISTS = (1, 3)
_GLCM_ANGLES = ((0, 1), (-1, 1), (-1, 0), (-1, -1))   # (dy,dx)*d  => 0,45,90,135 graus
_ORI_BINS = 12


def _names_color():
    n = []
    n += [f"rgb_{s}_{c}" for s in ("mean", "std") for c in "rgb"]
    n += [f"rgb_hist_{c}{i}" for c in "rgb" for i in range(_RGB_BINS)]
    n += ["hsv_hue_cos", "hsv_hue_sin", "hsv_s_mean", "hsv_s_std", "hsv_v_mean", "hsv_v_std"]
    n += [f"hsv_hist_h{i}" for i in range(_H_BINS)] + [f"hsv_hist_s{i}" for i in range(_S_BINS)]
    n += [f"lab_{s}_{c}" for s in ("mean", "std") for c in ("L", "a", "b")]
    return n


def _names_texture():
    n = []
    for P, R in _LBP_CFG:
        n += [f"lbp_P{P}R{R}_{i}" for i in range(P + 2)]
    for d in _GLCM_DISTS:
        n += [f"glcm_d{d}_{p}" for p in ("contrast", "homogeneity", "energy", "correlation", "entropy")]
    return n


def _names_grad(size: int):
    n = [f"hog_{i}" for i in range(_hog_dim())]
    n += ["edge_density", "edge_gradmag_mean", "edge_gradmag_std"]
    n += [f"edge_ori{i}" for i in range(_ORI_BINS)]
    return n


def _names_shape():
    return (["shp_area_ratio", "shp_perimeter_ratio", "shp_circularity", "shp_aspect_minrect",
             "shp_aspect_moments", "shp_solidity", "shp_border_contact"]
            + [f"shp_hu{i}" for i in range(1, 8)])


def _hog_dim():
    return 3 * 3 * 4 * 9                            # 324 (ver _hog_descriptor)


def feature_names(include_shape: bool, size: int = 128):
    n = _names_color() + _names_texture() + _names_grad(size)
    if include_shape:
        n += _names_shape()
    return n


# ----------------------------- cor ------------------------------------------ #
def color_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    px = img[mask].astype(np.float32)               # (N,3) BGR
    ph = hsv[mask]
    pl = lab[mask].astype(np.float32)
    rgb = px[:, ::-1]                               # -> RGB
    f = []
    f += list(rgb.mean(0) / 255.0) + list(rgb.std(0) / 255.0)
    # histogramas RGB (8 bins por canal, normalizados)
    rgb_u8 = px[:, ::-1].astype(np.uint8)
    for c in range(3):
        h = np.bincount(rgb_u8[:, c] >> 5, minlength=_RGB_BINS).astype(np.float32)
        f += list(h / h.sum())
    # HSV: matiz é circular (0..179 no OpenCV => 0..358 graus). Usa média vetorial.
    ang = ph[:, 0].astype(np.float32) * (2.0 * math.pi / 180.0)
    f += [float(np.cos(ang).mean()), float(np.sin(ang).mean())]
    s = ph[:, 1].astype(np.float32) / 255.0
    v = ph[:, 2].astype(np.float32) / 255.0
    f += [float(s.mean()), float(s.std()), float(v.mean()), float(v.std())]
    hh = np.bincount((ph[:, 0].astype(np.int32) * _H_BINS) // 180, minlength=_H_BINS).astype(np.float32)
    f += list(hh[:_H_BINS] / hh.sum())
    sh = np.bincount(ph[:, 1] >> 5, minlength=_S_BINS).astype(np.float32)
    f += list(sh / sh.sum())
    f += list(pl.mean(0) / 255.0) + list(pl.std(0) / 255.0)
    return np.asarray(f, np.float32)


# ----------------------------- textura -------------------------------------- #
def _glcm_props(P: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    tot = P.sum()
    if tot <= 0:
        return np.zeros(5, np.float32)
    P = P / tot
    diff2 = (I - J) ** 2
    contrast = float((P * diff2).sum())
    homog = float((P / (1.0 + diff2)).sum())
    asm = float((P * P).sum())
    px = P.sum(1)
    idx = np.arange(P.shape[0], dtype=np.float64)
    mu = float((idx * px).sum())
    var = float((((idx - mu) ** 2) * px).sum())
    corr = float((P * (I - mu) * (J - mu)).sum() / var) if var > 1e-12 else 1.0
    nz = P[P > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    return np.array([contrast, homog, math.sqrt(asm), corr, ent], np.float32)


_I, _J = np.meshgrid(np.arange(_GLCM_LEVELS, dtype=np.float64),
                     np.arange(_GLCM_LEVELS, dtype=np.float64), indexing="ij")


def glcm_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GLCM mascarada, 32 níveis, 4 ângulos promediados (invariância à rotação)."""
    q = (gray >> 3).astype(np.int32)                # 256 -> 32 níveis
    H, W = q.shape
    out = []
    for d in _GLCM_DISTS:
        acc = np.zeros(5, np.float64)
        for dy, dx in _GLCM_ANGLES:
            oy, ox = dy * d, dx * d
            y0, y1 = max(0, -oy), H - max(0, oy)
            x0, x1 = max(0, -ox), W - max(0, ox)
            a = q[y0:y1, x0:x1]
            b = q[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
            ok = mask[y0:y1, x0:x1] & mask[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
            idx = (a[ok] * _GLCM_LEVELS + b[ok])
            P = np.bincount(idx, minlength=_GLCM_LEVELS ** 2).reshape(_GLCM_LEVELS, _GLCM_LEVELS).astype(np.float64)
            P = P + P.T                             # simétrica
            acc += _glcm_props(P, _I, _J)
        out += list(acc / len(_GLCM_ANGLES))
    return np.asarray(out, np.float32)


def lbp_maps(gray: np.ndarray):
    """Mapas LBP (uniforme) calculados UMA vez por imagem e reutilizados nas variantes."""
    return [local_binary_pattern(gray, P, R, method="uniform") for P, R in _LBP_CFG]


def lbp_features(maps, inner: np.ndarray) -> np.ndarray:
    f = []
    for (P, R), lbp in zip(_LBP_CFG, maps):
        v = lbp[inner].astype(np.int64)
        h = np.bincount(v, minlength=P + 2).astype(np.float32)
        f += list(h / max(h.sum(), 1.0))
    return np.asarray(f, np.float32)


# ----------------------------- gradientes ----------------------------------- #
_HOG = None


class _NumpyHOG:
    """HOG em NumPy/OpenCV-core (fallback quando cv2.HOGDescriptor não existe, p.ex. OpenCV 5.x).

    Mesma configuração do descritor do OpenCV: 9 bins (orientação não direcionada), células de
    size/4 px, blocos 2x2 células com passo de 1 célula, normalização L2-Hys => 324 dimensões.
    Interface compatível: .winSize e .compute(img) -> array (324, 1).
    """

    def __init__(self, size: int):
        self.winSize = (size, size)
        self.size = size
        self.cell = size // 4
        self.ncell = 4
        self.nbins = 9

    def compute(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32)
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ang = ang % 180.0
        b = np.minimum((ang * (self.nbins / 180.0)).astype(np.int32), self.nbins - 1)
        S, c, n = self.size, self.cell, self.ncell
        yy, xx = np.indices((S, S))
        cell_id = (np.minimum(yy // c, n - 1) * n + np.minimum(xx // c, n - 1))
        idx = (cell_id * self.nbins + b).ravel()
        hist = np.bincount(idx, weights=mag.ravel(), minlength=n * n * self.nbins)
        hist = hist.reshape(n, n, self.nbins).astype(np.float32)
        blocks = []
        for by in range(n - 1):
            for bx in range(n - 1):
                v = hist[by:by + 2, bx:bx + 2].ravel()
                v = v / np.sqrt((v ** 2).sum() + 1e-6)          # L2
                v = np.minimum(v, 0.2)                          # Hys: corta em 0,2
                v = v / np.sqrt((v ** 2).sum() + 1e-6)          # renormaliza
                blocks.append(v)
        return np.concatenate(blocks).astype(np.float32).reshape(-1, 1)


def _hog_descriptor(size: int):
    """HOG do OpenCV (9 bins; célula = size/4; bloco 2x2; passo 1 célula => 324 dims) ou fallback NumPy."""
    import os
    if hasattr(cv2, "HOGDescriptor") and not os.environ.get("AV2_FORCE_NUMPY_HOG"):
        c = size // 4
        return cv2.HOGDescriptor((size, size), (2 * c, 2 * c), (c, c), (c, c), 9)
    return _NumpyHOG(size)


def gradient_features(gray: np.ndarray, gray_for_hog: np.ndarray, inner: np.ndarray, hog) -> np.ndarray:
    hg = hog.compute(gray_for_hog).ravel().astype(np.float32)
    gf = gray.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mv, av = mag[inner], ang[inner] % 180.0         # orientação não-direcionada
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150) > 0
    dens = float(edges[inner].mean())
    ob = np.minimum((av * _ORI_BINS / 180.0).astype(np.int32), _ORI_BINS - 1)
    oh = np.bincount(ob, weights=mv, minlength=_ORI_BINS).astype(np.float32)
    oh = oh / max(oh.sum(), 1e-6)
    return np.concatenate([hg, np.array([dens, mv.mean() / 255.0, mv.std() / 255.0], np.float32), oh]).astype(np.float32)


# ----------------------------- forma ---------------------------------------- #
def shape_features(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    s = m.shape[0]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(len(_names_shape()), np.float32)
    area = float(m.sum())
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    circ = 4.0 * math.pi * area / (perim ** 2 + 1e-9)
    pts = np.vstack(cnts)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    asp_rect = max(w, h) / max(min(w, h), 1e-6)
    hull = cv2.convexHull(pts)
    solidity = area / max(cv2.contourArea(hull), 1e-6)
    mo = cv2.moments(m, binaryImage=True)
    if mo["m00"] > 0:
        mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
        common = math.sqrt(max(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2, 0.0))
        l1, l2 = (mu20 + mu02) / 2.0 + common, (mu20 + mu02) / 2.0 - common
        asp_mom = math.sqrt(max(l1, 1e-9) / max(l2, 1e-9))
    else:
        asp_mom = 1.0
    border = (m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum()) / (4.0 * s)
    hu = cv2.HuMoments(mo).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    v = [area / (s * s), perim / (4.0 * s), min(circ, 1.5), min(asp_rect, 20.0),
         min(asp_mom, 20.0), solidity, border] + list(hu)
    return np.asarray(v, np.float32)


# =========================================================================== #
# 4) EXTRAÇÃO COMPLETA DE UMA IMAGEM (duas variantes em uma única leitura)
# =========================================================================== #
_erode_cache: dict = {}


def _inner_mask(mask: np.ndarray, r: int = 3) -> np.ndarray:
    """Erode a máscara r pixels p/ evitar efeitos de borda (LBP, gradientes)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    e = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    if e.sum() < 0.01 * e.size:                      # folha muito pequena: não erodir
        e = mask.copy()
    return e


def _border_exclusion(size: int, r: int = 3) -> np.ndarray:
    m = np.zeros((size, size), bool)
    m[r:-r, r:-r] = True
    return m


def extract_variants(path, params: PreprocParams, seg_factor: float,
                     do_full: bool = True, do_seg: bool = True):
    """Lê uma imagem e devolve (vec_full, vec_seg, meta).

    vec_full : características da imagem inteira (sem forma);
    vec_seg  : características restritas à máscara da folha (+ forma);
    meta     : dict com flags (corrompida, seg_ok, área da máscara).
    Retorna (None, None, meta) se a imagem não puder ser lida.
    """
    global _HOG
    img0 = read_image(path)
    if img0 is None or img0.size == 0:
        return None, None, {"read_ok": False}
    img = preprocess(img0, params)
    S = params.size
    if _HOG is None or _HOG.winSize[0] != S:
        _HOG = _hog_descriptor(S)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    meta = {"read_ok": True}
    vf = vs = None
    lmaps = lbp_maps(gray)

    if do_full:
        full = np.ones((S, S), bool)
        inner = _border_exclusion(S)
        vf = np.concatenate([
            color_features(img, full),
            np.concatenate([lbp_features(lmaps, inner), glcm_features(gray, full)]),
            gradient_features(gray, gray, inner, _HOG),
        ])
    if do_seg:
        mask, ok = segment_leaf(img, seg_factor)
        meta["seg_ok"] = bool(ok)
        meta["mask_area"] = float(mask.mean())
        inner = _inner_mask(mask)
        gray_masked = (gray * mask).astype(np.uint8)
        vs = np.concatenate([
            color_features(img, mask),
            np.concatenate([lbp_features(lmaps, inner), glcm_features(gray, mask)]),
            gradient_features(gray, gray_masked, inner, _HOG),
            shape_features(mask),
        ])
    return vf, vs, meta


# =========================================================================== #
# 5) EXTRAÇÃO EM LOTE (paralela)
# =========================================================================== #
def extract_chunk(paths, params_dict: dict, seg_factor: float, do_full: bool, do_seg: bool):
    """Processa uma lista de caminhos sequencialmente (executado em um processo-trabalhador)."""
    params = PreprocParams(**params_dict)
    nf = len(feature_names(False, params.size))
    ns = len(feature_names(True, params.size))
    n = len(paths)
    Xf = np.full((n, nf), np.nan, np.float32) if do_full else None
    Xs = np.full((n, ns), np.nan, np.float32) if do_seg else None
    read_ok = np.zeros(n, bool)
    seg_ok = np.zeros(n, bool)
    area = np.full(n, np.nan, np.float32)
    for i, p in enumerate(paths):
        vf, vs, meta = extract_variants(p, params, seg_factor, do_full, do_seg)
        if not meta.get("read_ok", False):
            continue
        read_ok[i] = True
        if do_full:
            Xf[i] = vf
        if do_seg:
            Xs[i] = vs
            seg_ok[i] = meta["seg_ok"]
            area[i] = meta["mask_area"]
    return Xf, Xs, read_ok, seg_ok, area


def extract_parallel(paths, params: PreprocParams, seg_factor: float, do_full: bool = True,
                     do_seg: bool = True, n_jobs: int = 1, chunk: int = 250):
    """Extrai características de muitas imagens em paralelo (lotes de `chunk` imagens)."""
    from joblib import Parallel, delayed
    paths = [str(p) for p in paths]
    parts = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
    res = Parallel(n_jobs=n_jobs)(delayed(extract_chunk)(c, params.as_dict(), seg_factor, do_full, do_seg)
                                  for c in parts)
    Xf = np.vstack([r[0] for r in res]) if do_full else None
    Xs = np.vstack([r[1] for r in res]) if do_seg else None
    return (Xf, Xs, np.concatenate([r[2] for r in res]), np.concatenate([r[3] for r in res]),
            np.concatenate([r[4] for r in res]))


def columns_for(names, prefixes):
    """Índices das colunas cujo nome começa com algum dos prefixos."""
    pref = tuple(prefixes)
    return [i for i, n in enumerate(names) if n.startswith(pref)]
