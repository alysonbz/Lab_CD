import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cell_images" / "cell_images"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# paleta categórica (ordem fixa por classe), tinta de texto e superfície
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def fmt_pct(p):
    return f"{p:.1%}".replace(".", ",")


def inspect(path):
    """Abre e decodifica a imagem. Levanta exceção se estiver corrompida."""
    with Image.open(path) as img:
        img.verify()  # estrutura do arquivo / CRC dos chunks
    with Image.open(path) as img:  # verify() invalida o objeto, reabre
        img.load()  # decodificação completa: pega arquivos truncados
        pixels = img.convert("RGB")
        digest = hashlib.md5(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
        return img.width, img.height, img.mode, digest


def scan():
    images, invalid = [], []
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.is_file())
        for f in tqdm(files, desc=class_dir.name, unit="img"):
            base = {"classe": class_dir.name, "arquivo": f.name}
            if f.suffix.lower() not in IMAGE_EXTS:
                invalid.append({**base, "tipo": "não é imagem", "detalhe": f.suffix})
                continue
            try:
                width, height, mode, digest = inspect(f)
            except Exception as e:  # PIL levanta OSError, SyntaxError, UnidentifiedImageError...
                invalid.append({**base, "tipo": "corrompida", "detalhe": f"{type(e).__name__}: {e}"})
                continue
            images.append({**base, "largura": width, "altura": height, "modo": mode, "hash": digest})
    images = pd.DataFrame(images)
    invalid = pd.DataFrame(invalid, columns=["classe", "arquivo", "tipo", "detalhe"])
    return images, invalid


def find_duplicates(images):
    dups = images[images.duplicated("hash", keep=False)].copy()
    dups["grupo"] = dups.groupby("hash").ngroup()
    dups["entre_classes"] = dups.groupby("hash")["classe"].transform("nunique") > 1
    return dups.sort_values(["grupo", "classe", "arquivo"])[
        ["grupo", "classe", "arquivo", "entre_classes", "hash"]
    ]


def class_summary(images, invalid, dups):
    summary = pd.DataFrame({
        "imagens_validas": images.groupby("classe").size(),
        "invalidas": invalid.groupby("classe").size(),
        "em_duplicatas": dups.groupby("classe").size(),
    }).fillna(0).astype(int)
    summary.index.name = "classe"
    summary.insert(0, "arquivos", summary["imagens_validas"] + summary["invalidas"])
    summary["proporcao"] = summary["imagens_validas"] / summary["imagens_validas"].sum()
    return summary


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    ax.set_facecolor(SURFACE)


def plot_balance(summary):
    classes = summary.index.tolist()
    counts = summary["imagens_validas"]
    props = summary["proporcao"]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

    fig, (ax_count, ax_prop) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.4]}, facecolor=SURFACE
    )

    bars = ax_count.bar(classes, counts, color=colors, width=0.5)
    ax_count.bar_label(bars, labels=[fmt_int(n) for n in counts], padding=4, color=INK)
    ax_count.set_ylim(0, counts.max() * 1.15)
    ax_count.set_ylabel("nº de imagens", color=INK_2)
    ax_count.set_title("Imagens por classe", loc="left", color=INK)
    ax_count.grid(axis="y", color=GRID, linewidth=0.8)
    ax_count.set_axisbelow(True)
    style(ax_count)

    left = 0.0
    for cls, p, color in zip(classes, props, colors):
        ax_prop.barh(0, p, left=left, color=color, height=0.45, edgecolor=SURFACE, linewidth=2)
        ax_prop.text(left + p / 2, 0, f"{cls}\n{fmt_pct(p)}", ha="center", va="center", color="white")
        left += p
    ax_prop.set_xlim(0, 1)
    ax_prop.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_prop.set_yticks([])
    ax_prop.spines["left"].set_visible(False)
    ax_prop.set_title("Proporção entre classes", loc="left", color=INK)
    style(ax_prop)

    fig.tight_layout()
    fig.savefig(OUT / "balanceamento_classes.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def report(images, invalid, dups, summary):
    n_groups = dups["hash"].nunique()
    cross = dups.loc[dups["entre_classes"], "hash"].nunique()
    ratio = summary["imagens_validas"].max() / summary["imagens_validas"].min()
    lines = [
        "== Análise do dataset cell_images ==",
        f"Pasta: {DATA}",
        f"Classes: {', '.join(summary.index)}",
        f"Arquivos lidos: {fmt_int(len(images) + len(invalid))}",
        f"Imagens válidas: {fmt_int(len(images))}",
        "",
        "-- Imagens por classe --",
        summary.to_string(formatters={"proporcao": fmt_pct}),
        f"Razão de desbalanceamento (maior/menor): {ratio:.3f}",
        "",
        "-- Arquivos inválidos --",
        invalid.groupby(["classe", "tipo"]).size().to_string() if len(invalid) else "nenhum",
        "",
        "-- Duplicatas (mesmo conteúdo de pixels) --",
        f"Grupos: {n_groups} | imagens envolvidas: {len(dups)} | cópias redundantes: {len(dups) - n_groups}",
        f"Grupos com cópias em classes diferentes (possível erro de rótulo): {cross}",
    ]
    text = "\n".join(lines)
    print(text)
    (OUT / "resumo.txt").write_text(text + "\n", encoding="utf-8")


def main():
    images, invalid = scan()
    dups = find_duplicates(images)
    summary = class_summary(images, invalid, dups)

    images.to_csv(OUT / "inventario.csv", index=False)
    invalid.to_csv(OUT / "arquivos_invalidos.csv", index=False)
    dups.to_csv(OUT / "duplicatas.csv", index=False)
    summary.to_csv(OUT / "contagem_por_classe.csv")
    plot_balance(summary)
    report(images, invalid, dups, summary)


if __name__ == "__main__":
    main()
