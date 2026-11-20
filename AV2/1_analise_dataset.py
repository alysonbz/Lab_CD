import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cell_images" / "cell_images"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
Q_LEVELS = 8  # cores quantizadas: 8 níveis por canal (faixas de 32 tons) -> 512 cores possíveis
TOP_COLORS = 12  # barras por painel no gráfico de paleta
MIN_SUPPORT = 2e-4  # cor "típica" precisa cobrir ao menos 0,02% dos pixels de célula da classe
MEAN_COLS = ["media_R", "media_G", "media_B"]  # cor média dos pixels da célula em cada imagem
N_SWATCHES = 400  # imagens sorteadas por classe nas faixas de cor média

# paleta categórica (ordem fixa por classe), tinta de texto e superfície
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def fmt_pct(p, decimals=1):
    return f"{p:.{decimals}%}".replace(".", ",")


def cell_pixels(pixels):
    """Pixels da célula como matriz (n, 3); o fundo, exatamente (0,0,0), fica de fora."""
    px = pixels.reshape(-1, 3)
    return px[px.max(axis=1) > 0]


class ColorStats:
    """Acumula as cores dos pixels da célula de uma classe (recebe só pixels de célula, ver cell_pixels)."""

    def __init__(self):
        n_bins = Q_LEVELS ** 3
        self.channels = np.zeros((3, 256), np.int64)  # histograma de cada canal R, G, B
        self.q_count = np.zeros(n_bins, np.int64)  # nº de pixels em cada cor quantizada
        self.q_sum = np.zeros((n_bins, 3))  # soma de R, G, B por cor quantizada (para a cor média)

    def add(self, px):
        q = px // (256 // Q_LEVELS)
        idx = (q[:, 0].astype(np.intp) * Q_LEVELS + q[:, 1]) * Q_LEVELS + q[:, 2]
        self.q_count += np.bincount(idx, minlength=self.q_count.size)
        for c in range(3):
            self.channels[c] += np.bincount(px[:, c], minlength=256)
            self.q_sum[:, c] += np.bincount(idx, weights=px[:, c], minlength=self.q_count.size)


def inspect(path):
    """Abre e decodifica a imagem. Levanta exceção se estiver corrompida. Devolve também os pixels."""
    with Image.open(path) as img:
        img.verify()  # estrutura do arquivo / CRC dos chunks
    with Image.open(path) as img:  # verify() invalida o objeto, reabre
        img.load()  # decodificação completa: pega arquivos truncados
        digest = hashlib.md5(f"{img.mode}{img.size}".encode() + img.tobytes()).hexdigest()
        return img.width, img.height, img.mode, digest, np.asarray(img)


def scan():
    images, invalid, colors = [], [], {}
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.is_file())
        for f in tqdm(files, desc=class_dir.name, unit="img"):
            base = {"classe": class_dir.name, "arquivo": f.name}
            if f.suffix.lower() not in IMAGE_EXTS:
                invalid.append({**base, "tipo": "não é imagem", "detalhe": f.suffix})
                continue
            try:
                width, height, mode, digest, pixels = inspect(f)
            except Exception as e:  # PIL levanta OSError, SyntaxError, UnidentifiedImageError...
                invalid.append({**base, "tipo": "corrompida", "detalhe": f"{type(e).__name__}: {e}"})
                continue
            row = {**base, "largura": width, "altura": height, "modo": mode, "hash": digest}
            if mode == "RGB":  # estatística de cor só faz sentido com 3 canais RGB
                px = cell_pixels(pixels)
                colors.setdefault(class_dir.name, ColorStats()).add(px)
                if len(px):  # cor média da célula nesta imagem
                    row.update(zip(MEAN_COLS, px.mean(axis=0)))
            images.append(row)
    images = pd.DataFrame(images)
    invalid = pd.DataFrame(invalid, columns=["classe", "arquivo", "tipo", "detalhe"])
    return images, invalid, colors


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


def color_table(colors):
    """Uma linha por cor quantizada presente: cor média, proporção dos pixels de célula em cada classe
    e quantas vezes ela é mais frequente na classe do que nas demais (razao_<classe>)."""
    total_count = sum(s.q_count for s in colors.values())
    total_sum = sum(s.q_sum for s in colors.values())
    used = np.flatnonzero(total_count)
    rgb = np.round(total_sum[used] / total_count[used, None]).astype(int)
    table = pd.DataFrame(rgb, columns=["r", "g", "b"], index=pd.Index(used, name="bin"))
    table.insert(0, "cor_hex", [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb])
    for cls, stats in colors.items():
        table[f"prop_{cls}"] = stats.q_count[used] / stats.q_count.sum()
    for cls in colors:
        others = [s for k, s in colors.items() if k != cls]
        if not others:
            continue
        p_rest = sum(s.q_count for s in others)[used] / sum(s.q_count.sum() for s in others)
        p_cls = table[f"prop_{cls}"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            table[f"razao_{cls}"] = np.where(p_rest > 0, p_cls / p_rest, np.inf)
    return table


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    ax.set_facecolor(SURFACE)


def bin_edges(values, step):
    lo = np.floor(values.min() / step) * step
    hi = max(np.ceil(values.max() / step) * step, lo + step)
    return np.arange(lo, hi + step / 2, step)


def int_bin_edges(values, target):
    """Bordas para valores inteiros. As dimensões só ocorrem de 3 em 3 px: com bins de largura que não é
    múltiplo desse passo, uns bins recebem 1 valor e outros 2, e o histograma fica serrilhado. A largura
    vira o múltiplo do passo mais próximo de target, com bordas entre os valores possíveis."""
    values = np.unique(values)
    grain = int(np.gcd.reduce(np.diff(values))) if len(values) > 1 else 1
    width = grain * max(1, round(target / grain))
    return np.arange(values.min() - grain / 2, values.max() + width, width)


def plot_balance(summary, class_colors):
    """Uma barra por classe com N e % do total no rótulo."""
    classes = summary.index.tolist()
    counts = summary["imagens_validas"]
    props = summary["proporcao"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    bars = ax.bar(classes, counts, color=[class_colors[c] for c in classes], width=0.5)
    ax.bar_label(bars, labels=[f"{fmt_int(n)} ({fmt_pct(p)})" for n, p in zip(counts, props)],
                 padding=4, color=INK, fontsize=11)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_ylabel("nº de imagens", color=INK_2)
    ax.set_title("Imagens por classe: N (% do total)", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)

    fig.tight_layout()
    fig.savefig(OUT / "balanceamento_classes.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_examples(n=8, seed=42):
    """Grade com n imagens aleatórias de cada classe; título = dimensões originais (largura x altura)."""
    rng = np.random.default_rng(seed)
    class_dirs = sorted(p for p in DATA.iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(class_dirs), n, figsize=(1.8 * n, 2.2 * len(class_dirs) + 0.5),
                             facecolor=SURFACE, squeeze=False)
    for row, class_dir in zip(axes, class_dirs):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for ax, i in zip(row, rng.choice(len(files), size=min(n, len(files)), replace=False)):
            with Image.open(files[i]) as img:
                ax.imshow(np.asarray(img))
                ax.set_title(f"{img.width}×{img.height}", fontsize=9, color=INK_2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(MUTED)
        row[0].set_ylabel(class_dir.name, color=INK, fontsize=11)
    fig.suptitle("Exemplos de imagens por classe (título: largura × altura em px)", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "exemplos_imagens.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dimensions(images, class_colors):
    """Histogramas de largura, altura (px) e proporção largura/altura, uma linha por classe."""
    ratio = images["largura"] / images["altura"]
    px_edges = int_bin_edges(images[["largura", "altura"]].to_numpy().ravel(), 5)
    panels = [
        ("Largura", images["largura"], px_edges, "largura (px)", "{:.0f} px"),
        ("Altura", images["altura"], px_edges, "altura (px)", "{:.0f} px"),
        ("Proporção largura / altura", ratio, bin_edges(ratio, 0.05), "razão (1 = quadrada)", "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (title, values, edges, xlabel, fmt) in zip(axes, panels):
        for cls, color in class_colors.items():
            v = values[images["classe"] == cls]
            counts, _ = np.histogram(v, bins=edges)
            median = fmt.format(v.median()).replace(".", ",")
            ax.stairs(counts, edges, color=color, linewidth=1.6, label=f"{cls} (mediana {median})")
        if title.startswith("Proporção"):
            ax.axvline(1, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel(xlabel, color=INK_2)
        ax.set_ylabel("nº de imagens", color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)
        style(ax)

    n_sizes = images.groupby(["largura", "altura"]).ngroups
    fig.suptitle(f"Dimensões das imagens: {fmt_int(n_sizes)} tamanhos distintos", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "dimensoes_imagens.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_channels(colors, class_colors):
    """Histograma de intensidade de cada canal RGB nos pixels da célula (fundo excluído).

    Escala log no eixo y: os tons escuros do parasita ocupam poucos pixels e somem na escala linear.
    """
    step = 4
    centers = np.arange(0, 256, step) + step / 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE, sharey=True)
    for c, (ax, name) in enumerate(zip(axes, ("R (vermelho)", "G (verde)", "B (azul)"))):
        for cls, stats in colors.items():
            hist = stats.channels[c].reshape(-1, step).sum(axis=1)
            share = np.where(hist > 0, hist / hist.sum(), np.nan)  # log(0) não existe: faixa vazia vira lacuna
            ax.plot(centers, share, drawstyle="steps-mid", color=class_colors[cls], linewidth=1.6, label=cls)
        ax.set_yscale("log")
        ax.set_xlim(0, 256)
        ax.set_title(f"Canal {name}", loc="left", color=INK)
        ax.set_xlabel("intensidade (0–255)", color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax)
    axes[0].set_ylabel("proporção dos pixels de célula (log)", color=INK_2)
    axes[0].legend(frameon=False, labelcolor=INK_2)
    fig.suptitle("Intensidade por canal nos pixels da célula (fundo preto excluído)", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "histograma_canais_rgb.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def eta2(values, groups):
    """Fração da variância de values explicada por groups (soma dos quadrados entre grupos / total)."""
    grand = values.mean()
    g = values.groupby(groups)
    return (g.size() * (g.mean() - grand) ** 2).sum() / ((values - grand) ** 2).sum()


def plot_image_colors(images, class_colors, seed=42):
    """Cor média da célula em cada imagem (fundo excluído): quanto ela muda de uma imagem para outra.

    Topo: uma faixa por classe com a cor média de N_SWATCHES imagens sorteadas, ordenadas pelo matiz.
    Embaixo: distribuição da média de cada canal entre as imagens, por classe. η² da classe = fração da
    variância entre imagens explicada pela classe; o restante é variação de uma imagem para outra que
    não depende da classe (coloração, iluminação), e é isso que a normalização de cor por imagem remove.
    """
    data = images.dropna(subset=MEAN_COLS)
    classes = [c for c in class_colors if (data["classe"] == c).any()]
    data = data[data["classe"].isin(classes)]
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(15, 4.8 + 1.1 * len(classes)), facecolor=SURFACE)
    grid = fig.add_gridspec(len(classes) + 1, 3, height_ratios=[0.6] * len(classes) + [3])

    for r, cls in enumerate(classes):
        means = data.loc[data["classe"] == cls, MEAN_COLS].to_numpy() / 255
        means = means[rng.choice(len(means), size=min(N_SWATCHES, len(means)), replace=False)]
        hue = rgb_to_hsv(means)[:, 0]
        means = means[np.argsort((hue + 0.5) % 1)]  # matiz girado meia volta: o rosa (H ≈ 0 ≈ 1) não se parte
        ax = fig.add_subplot(grid[r, :])
        ax.imshow(means[None], aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{cls}: cor média da célula em {fmt_int(len(means))} imagens sorteadas, ordenadas pelo matiz",
                     loc="left", color=INK)

    axes = []
    for c, (col, name) in enumerate(zip(MEAN_COLS, ("R (vermelho)", "G (verde)", "B (azul)"))):
        ax = fig.add_subplot(grid[-1, c], sharey=axes[0] if axes else None)
        axes.append(ax)
        edges = bin_edges(data[col], 4)
        for cls in classes:
            v = data.loc[data["classe"] == cls, col]
            counts, _ = np.histogram(v, bins=edges)
            ax.stairs(counts / len(v), edges, color=class_colors[cls], linewidth=1.6, label=cls)
        title = f"Canal {name}"
        if len(classes) > 1:
            title += f"  ·  η² da classe = {fmt_pct(eta2(data[col], data['classe']))}"
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("média da célula na imagem (0–255)", color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax)
    axes[0].set_ylabel("proporção das imagens da classe", color=INK_2)
    axes[0].legend(frameon=False, labelcolor=INK_2)

    fig.suptitle("Cor média da célula em cada imagem (fundo preto excluído)\n"
                 "η² da classe: fração da variância entre imagens explicada pela classe "
                 "(o resto muda de imagem para imagem)", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "variacao_cor_imagens.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def color_bars(ax, sub, values, labels, title, room):
    """Barras horizontais preenchidas com a própria cor; rótulo direto com o valor.

    room: limite do eixo x em múltiplos da maior barra (espaço para o rótulo à direita dela).
    """
    y = np.arange(len(sub))
    hexes = sub["cor_hex"].tolist()
    ax.barh(y, values, color=hexes, height=0.72)
    ax.set_yticks(y, hexes, family="monospace", fontsize=11)
    ax.invert_yaxis()
    for yi, v, text in zip(y, values, labels):
        ax.text(v, yi, f" {text}", va="center", color=INK, fontsize=12)
    ax.set_xlim(0, values.max() * room)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=14)
    style(ax)


def fmt_ratio(r):
    if np.isinf(r):
        return "só nesta classe"
    return f"{r:.0f}×" if r >= 10 else f"{r:.1f}×".replace(".", ",")


def plot_palette(table, classes):
    """Por classe: cores mais frequentes e cores típicas (mais frequentes nela do que nas demais)."""
    has_ratio = len(classes) > 1
    fig, axes = plt.subplots(len(classes), 2 if has_ratio else 1, figsize=(13, 4.6 * len(classes) + 1.2),
                             facecolor=SURFACE, squeeze=False)
    for row, cls in zip(axes, classes):
        prop = f"prop_{cls}"
        top = table.sort_values(prop, ascending=False).head(TOP_COLORS)
        color_bars(row[0], top, top[prop].to_numpy(), [fmt_pct(p) for p in top[prop]],
                   f"{cls}: cores mais frequentes", room=1.3)
        if has_ratio:
            ratio = f"razao_{cls}"
            typical = (table[table[prop] >= MIN_SUPPORT]
                       .sort_values([ratio, prop], ascending=False).head(TOP_COLORS))
            labels = [f"{fmt_pct(p, 2)} ({fmt_ratio(r)})" for p, r in zip(typical[prop], typical[ratio])]
            color_bars(row[1], typical, typical[prop].to_numpy(), labels,
                       f"{cls}: cores típicas da classe", room=1.75)
    fig.suptitle(f"Cores quantizadas ({Q_LEVELS} níveis por canal) nos pixels da célula\n"
                 "valor = % dos pixels de célula da classe; × = quantas vezes mais frequente que nas demais classes",
                 color=INK, fontsize=14)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT / "paleta_cores.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def report(images, invalid, dups, summary):
    n_groups = dups["hash"].nunique()
    cross = dups.loc[dups["entre_classes"], "hash"].nunique()
    ratio = summary["imagens_validas"].max() / summary["imagens_validas"].min()
    dims = images[["largura", "altura"]].agg(["min", "median", "max"]).T
    n_sizes = images.groupby(["largura", "altura"]).ngroups
    square = (images["largura"] == images["altura"]).mean()
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
        invalid.to_string(index=False) if len(invalid) else "nenhum",
        "",
        "-- Duplicatas (mesmo conteúdo de pixels) --",
        f"Grupos: {n_groups} | imagens envolvidas: {len(dups)} | cópias redundantes: {len(dups) - n_groups}",
        f"Grupos com cópias em classes diferentes (possível erro de rótulo): {cross}",
        "",
        "-- Modo de cor --",
        images["modo"].value_counts().to_string(),
        "",
        "-- Dimensões (px) --",
        dims.to_string(float_format=lambda v: f"{v:.0f}"),
        f"Tamanhos distintos: {fmt_int(n_sizes)} | imagens quadradas: {fmt_pct(square)}",
    ]
    text = "\n".join(lines)
    print(text)
    (OUT / "resumo.txt").write_text(text + "\n", encoding="utf-8")


def main():
    images, invalid, colors = scan()
    dups = find_duplicates(images)
    summary = class_summary(images, invalid, dups)
    class_colors = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(summary.index)}

    plot_balance(summary, class_colors)
    plot_examples()
    plot_dimensions(images, class_colors)
    report(images, invalid, dups, summary)

    if colors:
        table = color_table(colors)
        table.to_csv(OUT / "cores_quantizadas.csv")
        plot_channels(colors, class_colors)
        plot_image_colors(images, class_colors)
        plot_palette(table, list(colors))


if __name__ == "__main__":
    main()
