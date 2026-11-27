"""
PASSO 1 / TÓPICO 1 - Análise e preparação do dataset PlantVillage.

Executa, nesta ordem:
  1. download do repositório oficial (git sparse-checkout: somente raw/color + leaf_grouping);
  2. verificação da estrutura, das classes e da coerência com a documentação;
  3. investigação de divisões oficiais (nada é presumido: tudo é verificado);
  4. varredura paralela (md5, integridade, dimensões) com cache;
  5. auditoria de rótulos (duplicatas exatas, conflitos, coerência com o leaf-map);
  6. agrupamento por folha (leaf_id) + estratégia alternativa validada para as
     classes sem mapeamento; união de grupos por duplicatas exatas;
  7. divisão treino/teste agrupada e estratificada (sem vazamento);
  8. gráficos/tabelas (distribuição, balanceamento, exemplos, dificuldades visuais).

Uso:  python step01_dataset_analysis.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

import config
from utils import (log, timer, savefig, save_table, require, photo_base, leaf_map_key,
                   parse_photo, split_class, pretty_class)

IMG_EXT = {".jpg", ".jpeg", ".png"}
BLOCK_CANDIDATES = (16, 32, 64, 128, 256)
MAX_FRAGMENTATION = 0.05          # taxa máx. aceitável de folhas reais fragmentadas pelo bloco


# =========================================================================== #
# 1. DOWNLOAD E VERIFICAÇÃO
# =========================================================================== #
def _run(cmd):
    log("   $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"Falha em {' '.join(cmd)}:\n{r.stderr[-800:]}")
    return r.stdout


def _download_zip_fallback():
    """Alternativa sem git: baixa o ZIP do GitHub e extrai apenas raw/color e leaf_grouping."""
    url = config.REPO_URL.replace(".git", "") + "/archive/refs/heads/master.zip"
    tmp = config.DIR_CACHE / "PlantVillage-master.zip"
    log(f"   baixando ZIP (grande, ~3 GB): {url}")
    urllib.request.urlretrieve(url, tmp)
    config.REPO_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp) as z:
        for m in z.infolist():
            rel = m.filename.split("/", 1)[-1]
            if rel.startswith(("raw/color/", "leaf_grouping/")) or rel in ("leaf-map.json", "README.md"):
                if m.is_dir():
                    continue
                dst = config.REPO_DIR / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
    tmp.unlink(missing_ok=True)


def _clear_git_locks(repo):
    """Remove arquivos *.lock deixados por execuções interrompidas do git (seguro: o script é o único usuário)."""
    gd = repo / ".git"
    if not gd.exists():
        return
    for lk in gd.rglob("*.lock"):
        try:
            lk.unlink()
            log(f"   trava antiga do git removida: {lk.name}")
        except Exception as e:
            log(f"   não foi possível remover {lk}: {e}")


def _disk_count_color() -> int:
    n = 0
    for _, _, files in os.walk(config.RAW_COLOR_DIR):
        n += len(files)
    return n


def _manifest_count_color(git) -> int:
    out = _run([git, "-C", str(config.REPO_DIR), "ls-tree", "-r", "--name-only", "HEAD"])
    return sum(1 for ln in out.splitlines() if ln.startswith("raw/color/"))


def _complete_checkout(git):
    """Garante que TODOS os arquivos de raw/color do commit estejam no disco.

    Um clone/checkout interrompido deixa só parte das classes; comparamos a contagem de arquivos
    em disco com a do manifesto do repositório (git ls-tree) e completamos o que faltar.
    """
    repo = config.REPO_DIR
    if not (repo / ".git").exists():
        return
    exp = _manifest_count_color(git)
    act = _disk_count_color() if config.RAW_COLOR_DIR.exists() else 0
    log(f"   arquivos em raw/color: {act} em disco / {exp} no repositório")
    if act >= exp:
        return
    log("   checkout incompleto: baixando os arquivos que faltam (pode levar alguns minutos)...")
    _clear_git_locks(repo)
    g = [git, "-c", "core.longpaths=true", "-C", str(repo)]
    _run(g + ["sparse-checkout", "set", "raw/color", "leaf_grouping"])
    _clear_git_locks(repo)
    _run(g + ["checkout", "HEAD", "--", "raw/color", "leaf_grouping"])
    act = _disk_count_color()
    log(f"   arquivos em raw/color após completar: {act} / {exp}")
    require(act >= exp, f"Download incompleto ({act}/{exp} arquivos). Verifique espaço em disco, antivírus e "
                        "'git config --global core.longpaths true'; apague a pasta data\\PlantVillage-Dataset e rode de novo.")


def _git_clone_and_sparse(git):
    repo = config.REPO_DIR
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        _run([git, "-c", "core.longpaths=true", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
              config.REPO_URL, str(repo)])
    _clear_git_locks(repo)
    _run([git, "-c", "core.longpaths=true", "-C", str(repo), "sparse-checkout", "set", "raw/color", "leaf_grouping"])
    _complete_checkout(git)


def ensure_dataset():
    rc = config.RAW_COLOR_DIR
    git = shutil.which("git")
    if rc.exists() and any(rc.iterdir()):
        if git:
            try:
                _complete_checkout(git)            # detecta e completa downloads parciais
            except SystemExit:
                raise
            except Exception as e:
                log(f"   aviso: não foi possível verificar a completude via git ({str(e)[:150]})")
        log(f"Dataset presente em {config.REPO_DIR}")
        return
    if git:
        try:
            _git_clone_and_sparse(git)
        except SystemExit:
            raise
        except Exception as e:
            # repositório parcial/corrompido de uma execução anterior: apaga e tenta uma vez mais
            log(f"   falha no git ({str(e)[:200]}). Removendo {config.REPO_DIR} e clonando novamente...")
            shutil.rmtree(config.REPO_DIR, ignore_errors=True)
            _git_clone_and_sparse(git)
    else:
        log("git não encontrado: usando download do ZIP.")
        _download_zip_fallback()
    require(rc.exists() and any(rc.iterdir()),
            f"Download incompleto: {rc} não existe ou está vazio. Verifique conexão/git.")


def repo_manifest() -> list[str]:
    """Lista de caminhos do repositório (inclusive pastas não baixadas), via git ls-tree."""
    git = shutil.which("git")
    if git and (config.REPO_DIR / ".git").exists():
        try:
            out = _run([git, "-C", str(config.REPO_DIR), "ls-tree", "-r", "--name-only", "HEAD"])
            return out.splitlines()
        except Exception as e:
            log(f"   ls-tree falhou ({e}); usando listagem local")
    return [str(p.relative_to(config.REPO_DIR)).replace("\\", "/")
            for p in config.REPO_DIR.rglob("*") if p.is_file()]


def _fetch_text(url: str, timeout: int = 25) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


# =========================================================================== #
# 2. VARREDURA PARALELA DAS IMAGENS
# =========================================================================== #
def _scan_chunk(paths):
    out = []
    for p in paths:
        md5 = ""
        ok, w, h, ch, err = True, 0, 0, 0, ""
        try:
            with open(p, "rb") as fh:
                data = fh.read()
            md5 = hashlib.md5(data).hexdigest()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    w, h = im.size
                    ch = len(im.getbands())
            except Exception as e:                      # arquivo corrompido/truncado
                ok, err = False, type(e).__name__
            size = len(data)
        except Exception as e:
            ok, err, size = False, type(e).__name__, 0
        out.append((md5, size, w, h, ch, ok, err))
    return out


def list_image_files():
    rc = config.RAW_COLOR_DIR
    rows = []
    for cdir in sorted(p for p in rc.iterdir() if p.is_dir()):
        for f in sorted(os.listdir(cdir)):
            rows.append((cdir.name, f))
    return rows


def build_inventory(force: bool = False) -> pd.DataFrame:
    rows = list_image_files()
    if config.INVENTORY_CSV.exists() and not force:
        inv = pd.read_csv(config.INVENTORY_CSV, keep_default_na=False)
        disk_ids = {f"{c}/{f}" for c, f in rows}
        # o cache só vale se corresponder EXATAMENTE aos arquivos atualmente em disco
        if "md5" in inv.columns and len(inv) == len(rows) and set(inv["image_id"]) == disk_ids:
            log(f"Inventário em cache: {len(inv)} imagens ({config.INVENTORY_CSV.name})")
            return inv
        log(f"   cache do inventário não corresponde aos arquivos em disco "
            f"({len(inv)} no cache x {len(rows)} em disco): refazendo a varredura")
    log(f"{len(rows)} arquivos encontrados em raw/color; varrendo (md5 + integridade) ...")
    paths = [str(config.RAW_COLOR_DIR / c / f) for c, f in rows]
    chunks = [paths[i:i + 400] for i in range(0, len(paths), 400)]
    res = Parallel(n_jobs=config.N_JOBS)(delayed(_scan_chunk)(c) for c in chunks)
    flat = [t for r in res for t in r]
    inv = pd.DataFrame(flat, columns=["md5", "size_bytes", "width", "height", "channels", "read_ok", "read_error"])
    inv.insert(0, "filename", [f for _, f in rows])
    inv.insert(0, "class", [c for c, _ in rows])
    inv.insert(0, "image_id", [f"{c}/{f}" for c, f in rows])
    inv["ext"] = inv["filename"].str.lower().str.extract(r"(\.[a-z0-9]+)$")[0].fillna("")
    inv["crop"] = inv["class"].map(lambda c: split_class(c)[0])
    inv["disease"] = inv["class"].map(lambda c: split_class(c)[1])
    return inv


# =========================================================================== #
# 3. AUDITORIA DE RÓTULOS E AGRUPAMENTO POR FOLHA
# =========================================================================== #
def load_leaf_map() -> dict:
    p = config.LEAF_MAP_JSON
    if not p.exists():
        alt = config.REPO_DIR / "leaf-map.json"
        require(alt.exists(), f"leaf-map.json não encontrado em {p} nem em {alt}.")
        p = alt
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_leaf_id(cls: str, filename: str, leaf_map: dict, aliases: dict | None = None):
    """leaf_id oficial (leaf-map.json) para a imagem, ou None se não mapeada.

    Retorna 'MISMATCH::<candidato>' quando a chave existe no leaf-map mas o rótulo do
    mapa não corresponde à pasta da imagem (e não há alias reconhecido).
    """
    aliases = aliases or {}
    cand = leaf_map.get(leaf_map_key(filename))
    if not cand:
        return None
    for c in cand:
        label = c.split(":::")[0]
        if cls in c or aliases.get(label) == cls:
            return c
    return "MISMATCH::" + cand[0]


def detect_aliases(inv_classes, filenames, leaf_map):
    """Detecta rótulos do leaf-map que NÃO são nomes de pasta, mas que apontam consistentemente
    para uma única pasta (>=95% dos casos). Ex.: 'Apple_Frogeye Spot' -> 'Apple___Black_rot'."""
    folders = set(inv_classes)
    votes = defaultdict(Counter)
    for c, f in zip(inv_classes, filenames):
        cand = leaf_map.get(leaf_map_key(f))
        if not cand or any(c in x for x in cand):
            continue
        label = cand[0].split(":::")[0]
        if label not in folders:
            votes[label][c] += 1
    aliases = {}
    for label, cnt in votes.items():
        cls, n = cnt.most_common(1)[0]
        if n / sum(cnt.values()) >= 0.95:
            aliases[label] = cls
    return aliases


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def choose_block_width(inv: pd.DataFrame):
    """Valida, nos dados COM leaf_id, o quanto o agrupamento por blocos fragmenta folhas reais."""
    m = inv[inv["leaf_id"].notna() & inv["num"].notna()]
    rows = []
    for B in BLOCK_CANDIDATES:
        blk = (m["class"] + "::" + m["prefix"] + "::" + (m["num"] // B).astype(int).astype(str))
        n_blocks = blk.groupby(m["leaf_id"]).nunique()
        rows.append({"largura_bloco": B,
                     "folhas_avaliadas": int(len(n_blocks)),
                     "folhas_fragmentadas": int((n_blocks > 1).sum()),
                     "taxa_fragmentacao": float((n_blocks > 1).mean())})
    tab = pd.DataFrame(rows)
    ok = tab[tab["taxa_fragmentacao"] <= MAX_FRAGMENTATION]
    B = int(ok["largura_bloco"].iloc[0]) if len(ok) else int(tab["largura_bloco"].iloc[-1])
    return B, tab


def assign_groups(inv: pd.DataFrame, leaf_map: dict):
    inv = inv.copy()
    inv["base"] = inv["filename"].map(photo_base)
    pn = inv["base"].map(parse_photo)
    inv["prefix"] = [p[0] for p in pn]
    inv["num"] = [p[1] for p in pn]
    aliases = detect_aliases(inv["class"], inv["filename"], leaf_map)
    if aliases:
        log(f"   aliases leaf-map -> pasta detectados: {aliases}")
    inv["leaf_id"] = [resolve_leaf_id(c, f, leaf_map, aliases) for c, f in zip(inv["class"], inv["filename"])]
    inv.attrs["aliases"] = aliases
    inv["leaf_map_mismatch"] = inv["leaf_id"].fillna("").str.startswith("MISMATCH::")
    inv.loc[inv["leaf_map_mismatch"], "leaf_id"] = None

    B, val_tab = choose_block_width(inv)
    log(f"   largura de bloco escolhida (fallback): {B}")

    inv["group_raw"] = [
        ("leaf::" + li) if isinstance(li, str) else
        (f"blk::{c}::{pf}::{int(n) // B}" if (n is not None and not pd.isna(n)) else f"img::{c}::{b}")
        for li, c, pf, n, b in zip(inv["leaf_id"], inv["class"], inv["prefix"], inv["num"], inv["base"])]
    inv["group_type"] = inv["group_raw"].str.split("::").str[0].map(
        {"leaf": "leaf-map (oficial)", "blk": "bloco de fotos consecutivas", "img": "imagem isolada"})

    # duplicatas exatas (md5) sempre ficam no mesmo grupo (união de grupos)
    uf = UnionFind()
    for g in inv["group_raw"]:
        uf.find(g)
    first_g = {}
    for md5, g in zip(inv["md5"], inv["group_raw"]):
        if md5 in first_g:
            uf.union(first_g[md5], g)
        else:
            first_g[md5] = g
    inv["group"] = [uf.find(g) for g in inv["group_raw"]]
    codes, _ = pd.factorize(inv["group"])
    inv["group_id"] = codes
    return inv, B, val_tab


def audit_labels(inv: pd.DataFrame):
    """Duplicatas exatas, conflitos de rótulo e coerência com leaf-map. Retorna (inv, resumo)."""
    s = {}
    s["n_arquivos"] = int(len(inv))
    s["n_nao_imagem"] = int((~inv["ext"].isin(IMG_EXT)).sum())
    s["n_corrompidas"] = int((~inv["read_ok"].astype(bool)).sum())
    s["extensoes"] = dict(Counter(inv["ext"]))
    s["dimensoes"] = {f"{w}x{h}": int(n) for (w, h), n in inv.groupby(["width", "height"]).size().items()}

    # duplicatas exatas
    dup = inv[inv.duplicated("md5", keep=False) & (inv["md5"] != "")]
    g = dup.groupby("md5")
    dup_tab = g.agg(n_copias=("image_id", "size"), n_classes=("class", "nunique"),
                    classes=("class", lambda x: " | ".join(sorted(set(x)))),
                    exemplos=("image_id", lambda x: " || ".join(list(x)[:3]))).reset_index()
    s["n_grupos_duplicados"] = int(len(dup_tab))
    s["n_imagens_em_duplicatas"] = int(len(dup))
    s["n_duplicatas_entre_classes"] = int((dup_tab["n_classes"] > 1).sum())

    # conflitos: mesma FOTO (nome-base) em classes diferentes
    pb = inv.assign(photo=inv["base"].str.lower())
    nc = pb.groupby("photo")["class"].nunique()
    conflict_photos = nc[nc > 1].index
    conf = pb[pb["photo"].isin(conflict_photos)]
    conf_tab = (conf.groupby("photo").agg(n_arquivos=("image_id", "size"),
                                          classes=("class", lambda x: " | ".join(sorted(set(x))))).reset_index())
    s["n_fotos_em_mais_de_uma_classe"] = int(len(conf_tab))
    s["n_leafmap_mismatch"] = int(inv["leaf_map_mismatch"].sum())
    # nomes-base repetidos (cópias aumentadas da mesma foto)
    rep = inv.groupby("base").size()
    s["n_fotos_unicas_por_nome"] = int(len(rep))
    s["n_arquivos_em_fotos_repetidas"] = int(rep[rep > 1].sum())

    # política de uso: remove conflitos de rótulo (md5 igual em classes distintas) e
    # mantém UMA cópia de duplicatas exatas dentro da mesma classe
    inv = inv.copy()
    inv["use"] = True
    inv["excluded_reason"] = ""
    bad = inv["read_ok"].astype(bool) == False
    inv.loc[bad, ["use", "excluded_reason"]] = [False, "corrompida"]
    cross = set(dup_tab.loc[dup_tab["n_classes"] > 1, "md5"])
    m = inv["md5"].isin(cross) & inv["use"]
    inv.loc[m, ["use", "excluded_reason"]] = [False, "duplicata exata com rotulos conflitantes"]
    still = inv[inv["use"]]
    dups = still.duplicated("md5", keep="first")
    inv.loc[still.index[dups.values], ["use", "excluded_reason"]] = [False, "duplicata exata (copia removida)"]
    s["n_excluidas"] = int((~inv["use"]).sum())
    s["excluidas_por_motivo"] = dict(Counter(inv.loc[~inv["use"], "excluded_reason"]))
    s["n_usadas"] = int(inv["use"].sum())
    return inv, s, dup_tab, conf_tab


# =========================================================================== #
# 4. DIVISÃO TREINO/TESTE (agrupada e estratificada)
# =========================================================================== #
def _split_with_seed(y, groups, seed):
    sgkf = StratifiedGroupKFold(n_splits=config.N_SPLIT_FOLDS, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, groups))
    return tr, te


def make_split(df: pd.DataFrame):
    """Escolhe, entre 10 sementes, a divisão com proporções por classe mais próximas de 20%.

    A escolha usa apenas rótulos e grupos (nenhuma informação de desempenho de modelo).
    """
    y = df["class"].to_numpy()
    groups = df["group_id"].to_numpy()
    classes = np.unique(y)
    best = None
    for seed in range(10):
        tr, te = _split_with_seed(y, groups, config.SEED + seed)
        te_c = pd.Series(y[te]).value_counts().reindex(classes).fillna(0).to_numpy()
        tot_c = pd.Series(y).value_counts().reindex(classes).to_numpy()
        dev = np.abs(te_c / tot_c - config.TEST_FRACTION).sum()
        if te_c.min() > 0 and (best is None or dev < best[0]):
            best = (dev, seed, tr, te)
    require(best is not None, "Nenhuma divisão conseguiu incluir todas as classes no teste.")
    dev, seed, tr, te = best
    split = np.array(["train"] * len(df), dtype=object)
    split[te] = "test"
    return split, config.SEED + seed, float(dev)


def simulate_fallback_leakage(inv: pd.DataFrame, B: int, seed: int):
    """Estima o vazamento residual da estratégia de blocos usando imagens COM leaf_id.

    Agrupa as imagens mapeadas apenas por bloco (ignorando o leaf_id), divide e conta
    quantas imagens de teste têm a mesma folha real no treino.
    """
    m = inv[inv["use"] & inv["leaf_id"].notna() & inv["num"].notna()].copy()
    blk = (m["class"] + "::" + m["prefix"] + "::" + (m["num"] // B).astype(int).astype(str))
    gid, _ = pd.factorize(blk)
    tr, te = _split_with_seed(m["class"].to_numpy(), gid, seed)
    leaves_tr = set(m["leaf_id"].iloc[tr])
    leaked = m["leaf_id"].iloc[te].isin(leaves_tr)
    return {"n_teste_simulado": int(len(te)), "n_teste_com_folha_no_treino": int(leaked.sum()),
            "fracao": float(leaked.mean())}


# =========================================================================== #
# 5. DIVISÕES OFICIAIS (investigação)
# =========================================================================== #
def evaluate_candidate_split(name, train_ids, test_ids, inv_use):
    ids_all = set(inv_use["image_id"])
    tr, te = set(train_ids) & ids_all, set(test_ids) & ids_all
    cover = (len(tr) + len(te)) / max(len(ids_all), 1)
    disjoint = len(tr & te) == 0
    gmap = dict(zip(inv_use["image_id"], inv_use["group"]))
    shared = len({gmap[i] for i in tr} & {gmap[i] for i in te})
    return {"candidato": name, "n_treino": len(tr), "n_teste": len(te),
            "cobertura_do_dataset": round(cover, 4), "treino_teste_disjuntos": disjoint,
            "grupos_compartilhados_treino_teste": shared,
            "frac_teste": round(len(te) / max(len(tr) + len(te), 1), 4)}


def _parse_mapping(txt: str):
    """Linhas 'raw/color/<classe>/<arq>\\tSVM/<train|test>/<n>/<uuid>.JPG'."""
    tr, te = [], []
    for ln in txt.splitlines():
        parts = ln.strip().split("\t")
        if len(parts) != 2 or not parts[0].startswith("raw/color/"):
            continue
        iid = parts[0][len("raw/color/"):]
        (tr if "/train/" in parts[1] else te).append(iid)
    return tr, te


def investigate_official_split(manifest, inv_use):
    """Procura por divisões oficiais SEM presumir nomes: lista candidatos e valida cada um."""
    base = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"
    cands = sorted({p.split("/")[0] + ("/" + p.split("/")[1] if p.count("/") >= 1 and
                    p.split("/")[0] == "data_distribution_for_SVM" else "")
                    for p in manifest if "split" in p.lower() or "distribution" in p.lower()
                    or p.lower().endswith(("train_mapping.txt", "test_mapping.txt"))})
    top = sorted({p.split("/")[0] for p in manifest})
    info = {"pastas_raiz_do_repositorio": top,
            "caminhos_candidatos_a_split": [c for c in cands if not c.startswith("raw")],
            "existe_pasta_splits_no_github": any(p.startswith("splits/") for p in manifest)}
    rows = []
    t_tr = _fetch_text(base + "data_distribution_for_SVM/train_mapping.txt")
    t_te = _fetch_text(base + "data_distribution_for_SVM/test_mapping.txt")
    if t_tr and t_te:
        # cada arquivo de mapping pode conter linhas de treino e de teste: junta os dois
        a1, b1 = _parse_mapping(t_tr)
        a2, b2 = _parse_mapping(t_te)
        rows.append(evaluate_candidate_split("data_distribution_for_SVM (GitHub)", a1 + a2, b1 + b2, inv_use))
    # Hugging Face (apenas arquivos de texto; NÃO usa load_dataset). Opcional/não fatal.
    hf = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_{}.txt"
    h_tr, h_te = _fetch_text(hf.format("train")), _fetch_text(hf.format("test"))
    info["splits_hf_acessiveis"] = bool(h_tr and h_te)
    if h_tr and h_te:
        norm = lambda t: [ln.strip().replace("raw/color/", "", 1) for ln in t.splitlines() if ln.strip()]
        rows.append(evaluate_candidate_split("splits/color_*.txt (Hugging Face)", norm(h_tr), norm(h_te), inv_use))
    return info, pd.DataFrame(rows)


# =========================================================================== #
# 6. GRÁFICOS E ANÁLISES VISUAIS
# =========================================================================== #
def plot_class_distribution(inv_use):
    cnt = inv_use["class"].value_counts().sort_values()
    crops = sorted(inv_use["crop"].unique())
    pal = dict(zip(crops, sns.color_palette("tab20", len(crops))))
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh([pretty_class(c) for c in cnt.index], cnt.values,
            color=[pal[split_class(c)[0]] for c in cnt.index])
    for i, v in enumerate(cnt.values):
        ax.text(v + 30, i, str(v), va="center", fontsize=7)
    ax.set_xlabel("número de imagens")
    ax.set_title(f"Distribuição de imagens por classe (N={cnt.sum()}, {len(cnt)} classes)")
    ax.axvline(cnt.mean(), ls="--", c="gray", lw=1, label=f"média = {cnt.mean():.0f}")
    ax.legend(loc="lower right")
    savefig(fig, "01_distribuicao_classes.png")
    cc = inv_use["crop"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(cc)), cc.values, color=[pal[c] for c in cc.index])
    ax.set_xticks(range(len(cc)))
    ax.set_xticklabels(cc.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("imagens")
    ax.set_title("Imagens por cultura (espécie)")
    savefig(fig, "01_distribuicao_culturas.png")


def balance_table(inv_use):
    cnt = inv_use["class"].value_counts()
    p = cnt / cnt.sum()
    tab = pd.DataFrame({"classe": cnt.index, "n_imagens": cnt.values, "percentual": (100 * p).round(2).values})
    ent = float(-(p * np.log2(p)).sum() / np.log2(len(p)))
    stats = {"n_classes": int(len(cnt)), "min": int(cnt.min()), "max": int(cnt.max()),
             "mediana": float(cnt.median()), "razao_max_min": float(cnt.max() / cnt.min()),
             "classe_majoritaria": cnt.index[0], "pct_majoritaria": float(100 * p.iloc[0]),
             "entropia_normalizada": ent,
             "acuracia_baseline_classe_majoritaria_pct": float(100 * p.iloc[0])}
    return tab, stats


def _thumb(path, s=112):
    im = cv2.imread(str(path))
    return cv2.cvtColor(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def plot_examples(inv_use, per_class=4):
    rng = np.random.default_rng(config.SEED)
    classes = sorted(inv_use["class"].unique())
    halves = [classes[i:i + 19] for i in range(0, len(classes), 19)]
    for k, cl in enumerate(halves):
        fig, axes = plt.subplots(len(cl), per_class, figsize=(per_class * 1.6, max(len(cl), 2) * 1.6), squeeze=False)
        for i, c in enumerate(cl):
            ids = inv_use.loc[inv_use["class"] == c, "image_id"].to_numpy()
            pick = rng.choice(ids, size=min(per_class, len(ids)), replace=False)
            for j in range(per_class):
                ax = axes[i, j]
                ax.axis("off")
                if j < len(pick):
                    ax.imshow(_thumb(config.RAW_COLOR_DIR / pick[j]))
                if j == 0:
                    ax.text(-0.05, 0.5, pretty_class(c), transform=ax.transAxes, ha="right",
                            va="center", fontsize=7)
        fig.suptitle(f"Exemplos por classe ({k + 1}/{len(halves)})", y=0.995)
        savefig(fig, f"01_exemplos_classes_{k + 1}.png", dpi=110)


def background_analysis(inv_use):
    """Quantifica a variação do fundo e testa se o FUNDO sozinho prediz a classe (atalho)."""
    rng = np.random.default_rng(config.SEED)
    rows = []
    for split, n in (("train", 100), ("test", 40)):
        for c, g in inv_use[inv_use["split"] == split].groupby("class"):
            idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
            for i in idx:
                r = g.iloc[i]
                rows.append((r["image_id"], c, split))
    sample = pd.DataFrame(rows, columns=["image_id", "class", "split"])

    def bg_feat(iid):
        im = cv2.imread(str(config.RAW_COLOR_DIR / iid))
        im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                               lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
        return np.concatenate([np.median(ring, 0), ring.std(0)])

    X = np.array(Parallel(n_jobs=config.N_JOBS, batch_size=64)(delayed(bg_feat)(i) for i in sample["image_id"]))
    tr, te = (sample["split"] == "train").to_numpy(), (sample["split"] == "test").to_numpy()
    y = sample["class"].to_numpy()
    rf = RandomForestClassifier(n_estimators=200, random_state=config.SEED, n_jobs=config.N_JOBS)
    rf.fit(X[tr], y[tr])
    pr = rf.predict(X[te])
    maj = pd.Series(y[tr]).value_counts().index[0]
    res = {"acc_fundo": float(accuracy_score(y[te], pr)),
           "f1macro_fundo": float(f1_score(y[te], pr, average="macro")),
           "acc_chance_uniforme": float(1.0 / len(np.unique(y))),
           "acc_classe_majoritaria": float((y[te] == maj).mean()),
           "n_treino_amostra": int(tr.sum()), "n_teste_amostra": int(te.sum())}
    crops = sample["class"].map(lambda c: split_class(c)[0]).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    uc = sorted(set(crops))
    pal = dict(zip(uc, sns.color_palette("tab20", len(uc))))
    for cr in uc:
        mk = crops == cr
        axes[0].scatter(X[mk, 1] - 128, X[mk, 2] - 128, s=6, alpha=.5, color=pal[cr], label=cr)
    axes[0].set_xlabel("a* do fundo (verde↔vermelho)")
    axes[0].set_ylabel("b* do fundo (azul↔amarelo)")
    axes[0].set_title("Cor do FUNDO das imagens (borda), por cultura")
    axes[0].legend(fontsize=5, ncol=2, markerscale=2)
    labs = ["acaso\n(uniforme)", "classe\nmajoritária", "RF só com\ncor do fundo"]
    vals = [res["acc_chance_uniforme"], res["acc_classe_majoritaria"], res["acc_fundo"]]
    axes[1].bar(labs, [100 * v for v in vals], color=["#bbb", "#999", "#d9534f"])
    for i, v in enumerate(vals):
        axes[1].text(i, 100 * v + 0.5, f"{100 * v:.1f}%", ha="center")
    axes[1].set_ylabel("acurácia no teste (%)")
    axes[1].set_title("O fundo sozinho prediz a classe? (atalho)")
    savefig(fig, "01_dificuldade_fundo.png")
    return res


def plot_split(inv_use):
    ct = pd.crosstab(inv_use["class"], inv_use["split"])
    ct["frac_teste"] = ct["test"] / ct.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    ct_s = ct.sort_values("frac_teste")
    ax.barh([pretty_class(c) for c in ct_s.index], 100 * ct_s["frac_teste"], color="#4c72b0")
    ax.axvline(100 * config.TEST_FRACTION, c="r", ls="--", label="alvo 20%")
    ax.set_xlabel("% das imagens da classe no TESTE")
    ax.set_title("Fração de teste por classe (divisão agrupada)")
    ax.legend()
    savefig(fig, "01_split_fracao_teste_por_classe.png")
    return ct


# =========================================================================== #
# MAIN
# =========================================================================== #
def main(force=False):
    summary = {}
    with timer("Download / verificação do dataset"):
        ensure_dataset()
        manifest = repo_manifest()
        (config.DIR_TAB / "manifesto_repositorio.txt").write_text(
            "\n".join(p for p in manifest if not p.startswith(("raw/", "data_distribution_for_SVM/train/",
                                                              "data_distribution_for_SVM/test/",
                                                              "generated_for_paper/", "logs/",
                                                              "leaf_grouping/leaf_maps/",
                                                              "leaf_grouping/filtered_leafmaps/"))),
            encoding="utf-8")
        classes_dirs = sorted(p.name for p in config.RAW_COLOR_DIR.iterdir() if p.is_dir())
        log(f"   classes (subpastas de raw/color): {len(classes_dirs)}")
        summary["n_pastas_classe"] = len(classes_dirs)
        n_seg = sum(1 for p in manifest if p.startswith("raw/segmented/"))
        n_gray = sum(1 for p in manifest if p.startswith("raw/grayscale/"))
        n_col = sum(1 for p in manifest if p.startswith("raw/color/"))
        summary["manifesto_contagem"] = {"color": n_col, "segmented": n_seg, "grayscale": n_gray}
        log(f"   manifesto do repositório: color={n_col} | segmented={n_seg} | grayscale={n_gray}")

    with timer("Inventário das imagens (md5, integridade, dimensões)"):
        inv = build_inventory(force)
        require(len(inv) > 0, "Nenhuma imagem encontrada.")
        crops = sorted(inv["crop"].unique())
        diseases = sorted({d for d in inv["disease"].unique() if d.lower() != "healthy"})
        pares = sorted({c for c in inv["class"].unique() if not c.lower().endswith("___healthy")})
        summary.update(n_arquivos=int(len(inv)), n_culturas=len(crops), n_doencas_distintas=len(diseases),
                       n_pares_cultura_doenca=len(pares),
                       documentacao_readme={"imagens": 54306, "culturas": 14, "doencas": 26, "classes": 38})
        log(f"   {len(inv)} arquivos | {len(crops)} culturas | {len(pares)} pares cultura-doença | "
            f"{len(diseases)} nomes de doença distintos | README: 54.306 imagens, 14 culturas, 26 doenças")

    with timer("Agrupamento por folha e auditoria de rótulos"):
        leaf_map = load_leaf_map()
        inv, B, val_tab = assign_groups(inv, leaf_map)
        save_table(val_tab, "01_validacao_largura_bloco.csv")
        inv, aud, dup_tab, conf_tab = audit_labels(inv)
        summary["auditoria"] = aud
        summary["largura_bloco"] = B
        summary["aliases_leaf_map"] = inv.attrs.get("aliases", {})
        summary["validacao_bloco"] = val_tab.to_dict("records")
        gt = inv["group_type"].value_counts()
        summary["imagens_por_tipo_de_grupo"] = {k: int(v) for k, v in gt.items()}
        log(f"   corrompidas={aud['n_corrompidas']} | duplicatas exatas (grupos)={aud['n_grupos_duplicados']} "
            f"(entre classes: {aud['n_duplicatas_entre_classes']}) | excluídas={aud['n_excluidas']} | usadas={aud['n_usadas']}")
        save_table(dup_tab, "01_duplicatas_exatas.csv")
        save_table(conf_tab, "01_fotos_em_mais_de_uma_classe.csv")
        src = (inv.assign(source=inv["prefix"].fillna("(sem número)"))
               .groupby(["source"]).agg(n=("image_id", "size"), n_classes=("class", "nunique"),
                                        classes=("class", lambda x: " | ".join(sorted(set(x))[:4])))
               .reset_index().sort_values("n", ascending=False))
        save_table(src, "01_prefixos_de_origem_das_fotos.csv")

    use = inv[inv["use"]].copy()
    with timer("Investigação de divisões oficiais"):
        info, cand_tab = investigate_official_split(manifest, use)
        summary["investigacao_split_oficial"] = info
        if len(cand_tab):
            save_table(cand_tab, "01_split_oficial_candidatos.csv")
            log("\n" + cand_tab.to_string(index=False))
        log(f"   pasta 'splits/' no GitHub: {info['existe_pasta_splits_no_github']} | "
            f"splits HF acessíveis: {info['splits_hf_acessiveis']}")

    with timer("Divisão treino/teste (agrupada + estratificada)"):
        split, seed, dev = make_split(use)
        use["split"] = split
        inv["split"] = ""
        inv.loc[use.index, "split"] = use["split"]
        leak = len(set(use.loc[use.split == "train", "group"]) & set(use.loc[use.split == "test", "group"]))
        require(leak == 0, f"VAZAMENTO: {leak} grupos presentes em treino e teste.")
        sim = simulate_fallback_leakage(inv, B, seed)
        n_tr, n_te = int((use.split == "train").sum()), int((use.split == "test").sum())
        fb_test = int(((use.split == "test") & (use.group_type != "leaf-map (oficial)")).sum())
        sim["n_teste_em_grupos_de_bloco"] = fb_test
        sim["vazamento_global_estimado_fracao_do_teste"] = sim["fracao"] * fb_test / max(n_te, 1)
        summary["split"] = {"semente": seed, "n_treino": n_tr, "n_teste": n_te,
                            "frac_teste": n_te / (n_tr + n_te), "desvio_total_proporcoes": dev,
                            "grupos_total": int(use["group_id"].nunique()),
                            "grupos_compartilhados": leak, "simulacao_vazamento_fallback": sim,
                            "metodo": "StratifiedGroupKFold(5), fold escolhido entre 10 sementes"}
        log(f"   treino={n_tr} teste={n_te} (frac={n_te / (n_tr + n_te):.3f}) | grupos compartilhados={leak} | "
            f"vazamento residual simulado (fallback)={100 * sim['fracao']:.2f}% | estimativa global "
            f"≈ {100 * sim['vazamento_global_estimado_fracao_do_teste']:.2f}% do teste")
        ct = plot_split(use)
        save_table(ct.reset_index(), "01_split_por_classe.csv")
        inv.to_csv(config.INVENTORY_CSV, index=False, encoding="utf-8")
        use[["image_id", "class", "group_id", "split"]].to_csv(config.SPLIT_CSV, index=False)

    with timer("Distribuição, balanceamento, exemplos e dificuldades visuais"):
        plot_class_distribution(use)
        tab, stats = balance_table(use)
        save_table(tab, "01_classes_contagem.csv")
        summary["balanceamento"] = stats
        log(f"   razão max/min={stats['razao_max_min']:.1f} | classe majoritária="
            f"{stats['classe_majoritaria']} ({stats['pct_majoritaria']:.1f}%) | entropia norm.={stats['entropia_normalizada']:.3f}")
        plot_examples(use)
        summary["fundo_atalho"] = background_analysis(use)
        log(f"   RF só com cor do fundo: acc={100 * summary['fundo_atalho']['acc_fundo']:.1f}% "
            f"(acaso {100 * summary['fundo_atalho']['acc_chance_uniforme']:.1f}%)")

    (config.DIR_REPORT / "resumo_passo1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("PASSO 1 concluído.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refaz a varredura das imagens")
    main(ap.parse_args().force)
