"""utils.py - funções auxiliares: log, tempo, memória, nomes de arquivo, figuras."""
from __future__ import annotations

import os
import re
import sys
import time
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")                      # gráficos salvos em arquivo (funciona sem display)
import matplotlib.pyplot as plt

import config

# console do Windows (cp1252) não suporta alguns símbolos: força UTF-8 sem quebrar
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

T0 = time.time()


def mem_mb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return None


def log(msg: str) -> None:
    m = mem_mb()
    ms = f" | RAM {m:,.0f} MB" if m else ""
    print(f"[{time.time() - T0:7.1f}s{ms}] {msg}", flush=True)


@contextmanager
def timer(name: str):
    t = time.time()
    log(f"▶ {name}")
    yield
    log(f"✔ {name} concluído em {time.time() - t:.1f}s")


def savefig(fig, name: str, dpi: int = 130) -> str:
    path = config.DIR_FIG / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"   figura salva: {path.name}")
    return str(path)


def save_table(df, name: str, index: bool = False) -> str:
    path = config.DIR_TAB / name
    df.to_csv(path, index=index, encoding="utf-8-sig")
    log(f"   tabela salva: {path.name}")
    return str(path)


# ----------------------------- nomes de arquivo ------------------------------ #
_UUID = re.compile(r"^[0-9a-fA-F-]{36}___")
_NUM = re.compile(r"^(.*?)[\s_]*(\d+)(?:\.\d+)?$")


def stem_no_ext(filename: str) -> str:
    return os.path.splitext(filename)[0]


def photo_base(filename: str) -> str:
    """Nome-base da FOTO original: sem UUID, sem sufixo 'copy', sem extensão.

    Várias imagens do PlantVillage são cópias aumentadas (giradas/espelhadas) da
    mesma foto e carregam o mesmo nome-base com UUIDs diferentes.
    """
    b = stem_no_ext(filename)
    b = _UUID.sub("", b)
    b = b.split("copy")[0].strip()
    return b


def leaf_map_key(filename: str) -> str:
    """Chave de busca no leaf-map.json (mesma lógica do loader oficial plant_village.py)."""
    x = filename.replace("_final_masked", "")
    if "___" in x:
        x = x.split("___")[-1]
    x = x.split("copy")[0]
    x = re.sub(r"\.(jpg|png|jpeg)$", "", x, flags=re.I).strip()
    return x.lower().strip()


def parse_photo(base: str):
    """Retorna (prefixo, numero) do nome-base, ou (None, None) se não houver número final."""
    m = _NUM.match(base)
    if not m:
        return None, None
    return m.group(1).strip(), int(m.group(2))


def split_class(cls: str):
    crop, _, disease = cls.partition("___")
    return crop, disease


def pretty_class(cls: str) -> str:
    crop, disease = split_class(cls)
    crop = crop.replace("_(including_sour)", "").replace("_(maize)", "").replace(",_bell", "")
    disease = disease.replace("_", " ").replace("(Citrus greening)", "").strip()
    return f"{crop}: {disease}"[:42]


def require(cond: bool, msg: str):
    """Interrompe a execução com mensagem clara (exigência do enunciado)."""
    if not cond:
        print(f"\n[ERRO] {msg}\n", file=sys.stderr)
        raise SystemExit(1)
