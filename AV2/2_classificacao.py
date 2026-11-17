"""Classificação do dataset cell_images: baseline, SVM e Regressão Logística.

Roda o pipeline abaixo para cada configuração em CONFIGS (redimensionamento, normalização de cor,
segmentação e conjunto de features) e compara os resultados em outputs/comparacao_preprocessamento.csv.

Uso: python 2_classificacao.py                 -> roda todas as configurações
     python 2_classificacao.py <config> [...]  -> roda só as indicadas; a comparação junta as métricas
                                                  já salvas das demais (de execuções anteriores)

1. extrai características de cor e textura de cada imagem (dataframe em outputs/features_*.csv;
   configurações que só diferem no conjunto de features reaproveitam o mesmo arquivo)
2. divide em treino/teste por paciente (StratifiedGroupKFold), sem paciente nos dois conjuntos;
   a divisão é a mesma em todas as configurações
3. ajusta hiperparâmetros por GridSearchCV com validação cruzada agrupada por paciente,
   somente no treino; o StandardScaler fica dentro do pipeline (ajustado só no treino de cada fold)
4. métricas e matrizes de confusão das predições out-of-fold -> outputs/<config>/validacao_cruzada/
5. avaliação final no conjunto de teste separado no passo 2 -> outputs/<config>/testset/,
   com as predições por imagem e exemplos de erros do melhor modelo
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern
from skimage.morphology import disk, erosion
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cell_images" / "cell_images"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# configurações comparadas
# redimensionamento: "64x64" = tamanho fixo; "fator2" = todas reduzidas à metade;
#                    "original" = sem redimensionar (ver downscale)
# normalizar: normalização de cor por imagem (normalize_color)
# segmentar: estatísticas só nos pixels da célula (cell_mask); False = imagem inteira, com fundo
# features: conjunto de colunas usado pelos modelos (FEATURE_SETS)
CONFIGS = {
    f"{size}_{norm_name}": {"redimensionamento": size, "normalizar": norm, "segmentar": True,
                            "features": "cor_textura"}
    for size in ("64x64", "fator2", "original")
    for norm_name, norm in (("sem_normalizacao", False), ("normalizacao_cor", True))
}
# configuração escolhida; as ablações abaixo partem dela. Redimensionamento: 64x64, fator2 e original
# empataram na validação cruzada (diferença < 0,002 no F1 macro, desvio entre folds ~0,01), então
# ficou o 64x64, o mais simples e rápido
FINAL = "64x64_normalizacao_cor"
CONFIGS[f"{FINAL}_so_cor"] = {**CONFIGS[FINAL], "features": "cor"}  # sem textura (LBP)
CONFIGS[f"{FINAL}_sem_segmentacao"] = {**CONFIGS[FINAL], "segmentar": False}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEED = 42
N_FOLDS = 5  # 1/5 dos pacientes vai para teste; mesmo nº de folds na validação cruzada
SIZE = 64  # lado da imagem no redimensionamento "64x64"
FACTOR = 2  # divisor do redimensionamento "fator2"
MASK_EROSION = 1  # raio (px) da erosão da máscara: remove a borda misturada com o fundo pelo resize
GRAY_TARGET = 0.6  # normalização de cor: média de cada canal na célula passa a valer isso (cinza neutro)
HIST_BINS = 16
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2  # padrões uniformes + "não uniforme"

FEATURE_NAMES = (
    [f"rgb_{stat}_{c}" for stat in ("media", "desvio") for c in "RGB"]
    + [f"hsv_{stat}_{c}" for stat in ("media", "desvio") for c in "SV"]
    + ["hsv_media_H_cos", "hsv_media_H_sin", "hsv_desvio_circ_H"]  # matiz: estatística circular (hue_stats)
    + [f"hist_{c}_{i}" for c in "RGB" for i in range(HIST_BINS)]
    + [f"lbp_{i}" for i in range(LBP_BINS)]
)
FEATURE_SETS = {
    "cor_textura": FEATURE_NAMES,
    "cor": [n for n in FEATURE_NAMES if not n.startswith("lbp_")],
}

# intervalos de teste dos hiperparâmetros (prefixo clf__ = passo do pipeline)
# SVM: os melhores pontos ficam na diagonal C·gamma ≈ 0,1 e o ótimo anterior estava no canto C=100,
# gamma=1e-3; a grade foi estendida nessa direção (C=1000, gamma=1e-4) e perdeu C=0,1 e gamma=0,1,
# sempre entre os piores nas duas configurações
MODELS = {
    "baseline": (  # referência: sempre prevê a classe mais frequente do treino
        "Baseline (classe mais frequente)",
        DummyClassifier(strategy="most_frequent"),
        {},
    ),
    "svm": (
        "SVM (RBF)",
        SVC(kernel="rbf", cache_size=1000),
        {"clf__C": np.logspace(0, 3, 4), "clf__gamma": np.logspace(-4, -2, 3)},
    ),
    "regressao_logistica": (
        "Regressão Logística",
        LogisticRegression(max_iter=5000),
        {"clf__C": np.logspace(-4, 2, 7)},
    ),
}

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CMAP = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def fmt_pct(p):
    return f"{p:.1%}".replace(".", ",")


def patient_id(filename):
    """C<n> no início do nome identifica o paciente/lâmina (ex.: C33P1thinF_IMG_... -> C33)."""
    match = re.match(r"C\d+", filename)
    return match.group() if match else filename.split("_IMG")[0]


def downscale(full, mode):
    """Redimensiona a imagem antes da extração (menos pixels -> extração mais rápida).

    "64x64": tamanho fixo. As imagens têm de 46 a 394 px, então cada uma é reduzida (ou ampliada) por
    um fator diferente e a proporção é distorcida: a mesma textura vira escalas diferentes no LBP.
    "fator2": todas reduzidas à metade (média 2x2). Como todas as células foram fotografadas com a
    mesma ampliação, 1 px continua valendo o mesmo tamanho físico em todas, e a proporção é preservada.
    "original": sem redimensionar. Mesma escala física e proporção, sem perder detalhe do parasita;
    as features são divididas por n, então não precisam de tamanho fixo. Extração ~2x mais lenta.
    """
    if mode == "original":
        return full
    if mode == "fator2":
        return full.reduce(FACTOR)
    return full.resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def cell_mask(full, size):
    """Máscara da célula: fundo é exatamente (0,0,0), então é separado na imagem original, antes do resize.

    A redução mistura fundo e célula na borda; a máscara é redimensionada com NEAREST (sem valores
    intermediários) para o mesmo tamanho da imagem reduzida e erodida para descartar esse anel misturado.
    """
    mask = Image.fromarray(np.asarray(full).max(axis=2) > 0)
    mask = np.asarray(mask.resize(size, Image.Resampling.NEAREST))
    return erosion(mask, disk(MASK_EROSION))


def normalize_color(rgb, mask):
    """Gray world dentro da célula: escala cada canal para que a média dos pixels da célula vire GRAY_TARGET.

    Remove o tom geral da lâmina (corante, espessura, iluminação), que varia entre lâminas e não depende
    da classe. O parasita ocupa poucos pixels, quase não altera a média e continua mais escuro/roxo que
    o resto da célula. Usa só a própria imagem: nada da classe nem de outras imagens.
    """
    mean = rgb[mask].mean(axis=0)
    return np.clip(rgb * (GRAY_TARGET / np.maximum(mean, 1e-6)), 0, 1)


def hue_stats(hue, sat):
    """Média e dispersão do matiz H por estatística circular, ponderada pela saturação.

    H é um ângulo (0 e 1 são o mesmo vermelho) e o rosa das células fica justamente nessa virada:
    a média aritmética de 0,98 e 0,02 daria 0,5 (ciano). Cada pixel vira o vetor (cos θ, sin θ),
    θ = 2πH, com peso = saturação, porque o matiz de um pixel quase cinza é instável.
    - média: 1º momento trigonométrico (C, S) = média ponderada de (cos θ, sin θ); a direção é o matiz
      médio e o comprimento R mede a concentração. Os componentes (e não o ângulo) evitam o salto 0/1 e
      ficam estáveis quando R ≈ 0 (célula quase cinza, como após a normalização de cor)
    - dispersão: desvio padrão circular sqrt(-2 ln R) (Mardia & Jupp, Directional Statistics)
    """
    total = sat.sum()
    if total <= 0:  # célula toda cinza: matiz indefinido
        return np.zeros(3)
    theta = 2 * np.pi * hue
    c = (sat * np.cos(theta)).sum() / total
    s = (sat * np.sin(theta)).sum() / total
    r = np.hypot(c, s)
    return np.array([c, s, np.sqrt(-2 * np.log(np.clip(r, 1e-12, 1)))])


def extract(path, resize_mode, normalize, segment=True):
    full = Image.open(path)  # todas as imagens já são RGB (ver outputs/resumo.txt): sem conversão
    small = downscale(full, resize_mode)
    rgb = np.asarray(small, dtype=np.float32) / 255
    mask = cell_mask(full, small.size) if segment else np.ones(rgb.shape[:2], bool)
    if mask.sum() < 10:  # célula minúscula ou imagem sem fundo detectável: usa tudo
        mask = np.ones_like(mask)
    if normalize:
        rgb = normalize_color(rgb, mask)
    gray = rgb2gray(rgb)
    n = mask.sum()

    px = rgb[mask]
    hsv = rgb2hsv(rgb)[mask]
    feats = [px.mean(axis=0), px.std(axis=0),  # R, G, B
             hsv[:, 1:].mean(axis=0), hsv[:, 1:].std(axis=0),  # S, V: escalas lineares
             hue_stats(hsv[:, 0], hsv[:, 1])]  # H: escala circular
    for c in range(3):
        hist, _ = np.histogram(rgb[..., c][mask], bins=HIST_BINS, range=(0, 1))
        feats.append(hist / n)
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=LBP_BINS, range=(0, LBP_BINS))
    feats.append(hist / n)
    return np.concatenate(feats)


def build_dataset(opts):
    rows, feats, skipped = [], [], []
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for f in tqdm(files, desc=f"features {class_dir.name}", unit="img"):
            try:
                feats.append(extract(f, opts["redimensionamento"], opts["normalizar"], opts["segmentar"]))
            except Exception as e:  # imagem corrompida: fica fora do dataset
                skipped.append(f"{class_dir.name}/{f.name}: {type(e).__name__}: {e}")
                continue
            rows.append({"id": f.name, "classe": class_dir.name, "paciente": patient_id(f.name)})
    features = pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)
    return pd.concat([pd.DataFrame(rows), features], axis=1), skipped


def split_train_test(df):
    """Primeiro fold de um StratifiedGroupKFold: ~20% para teste, estratificado e sem paciente repetido."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(df, df["classe"], df["paciente"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def describe_split(train, test):
    split = pd.concat([train.assign(conjunto="treino"), test.assign(conjunto="teste")], ignore_index=True)
    table = pd.crosstab(split["conjunto"], split["classe"], margins=True, margins_name="total")
    patients = split.groupby("conjunto")["paciente"].nunique()
    patients["total"] = split["paciente"].nunique()
    table["pacientes"] = patients
    print("\n-- Divisão treino/teste --")
    print(table.to_string())
    shared = set(train["paciente"]) & set(test["paciente"])
    print(f"Pacientes nos dois conjuntos: {len(shared)}")


def tune(estimator, grid, X, y, groups, cv):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    search.fit(X, y, groups=groups)
    return search


def grid_table(search):
    results = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in results if c.startswith("param_")]
    results = results[param_cols + ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]]
    results = results.rename(columns={c: c.removeprefix("param_clf__") for c in param_cols})
    return results.rename(columns={"mean_test_score": "f1_macro_media", "std_test_score": "f1_macro_desvio",
                                   "rank_test_score": "rank", "mean_fit_time": "tempo_fit_s"}).sort_values("rank")


def compute_metrics(name, y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"modelo": name, "classe": labels, "precisao": p, "recall": r, "f1": f, "suporte": s})
    overall = {"modelo": name, "acuracia": accuracy_score(y_true, y_pred),
               "precisao_macro": p.mean(), "recall_macro": r.mean(), "f1_macro": f.mean()}
    return per_class, overall


def plot_confusion(preds, y_true, labels, title, path):
    mats = {name: confusion_matrix(y_true, y_pred, labels=labels) for name, y_pred in preds.items()}
    vmax = max(m.max() for m in mats.values())  # mesma escala em todas as matrizes
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.8), facecolor=SURFACE, squeeze=False)
    for ax, (name, m) in zip(axes[0], mats.items()):
        ax.imshow(m, cmap=CMAP, vmin=0, vmax=vmax)
        row_pct = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        for i, j in np.ndindex(m.shape):
            ax.text(j, i, f"{fmt_int(m[i, j])}\n({fmt_pct(row_pct[i, j])})", ha="center", va="center",
                    fontsize=11, color="white" if m[i, j] > vmax / 2 else INK)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Classe predita", color=INK_2)
        ax.set_ylabel("Classe real", color=INK_2)
        ax.set_title(name, loc="left", color=INK)
        ax.tick_params(colors=INK_2, length=0)
        ax.spines[:].set_visible(False)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def save_results(folder, preds, y_true, labels, title, fig_name):
    """Métricas por classe e gerais em CSV e matrizes de confusão em fig_name (nome único por conjunto,
    para as figuras poderem ficar juntas numa mesma pasta, como no relatório)."""
    results = [compute_metrics(name, y_true, y_pred, labels) for name, y_pred in preds.items()]
    per_class = pd.concat([pc for pc, _ in results], ignore_index=True)
    overall = pd.DataFrame([ov for _, ov in results])
    per_class.to_csv(folder / "metricas_por_classe.csv", index=False)
    overall.to_csv(folder / "metricas_gerais.csv", index=False)
    plot_confusion(preds, y_true, labels, title, folder / fig_name)
    print(f"\n-- {title} --")
    print(per_class.round(4).to_string(index=False))
    print(overall.round(4).to_string(index=False))


def features_path(opts):
    """Arquivo de features de uma combinação de pré-processamento: outputs/<nome>/features_<nome>.csv.

    O conjunto de features não entra no nome: uma configuração que só difere nele (ex.: *_so_cor)
    reaproveita o arquivo da pasta da configuração com o mesmo pré-processamento.
    """
    norm = "normalizacao_cor" if opts["normalizar"] else "sem_normalizacao"
    seg = "" if opts["segmentar"] else "_sem_segmentacao"
    name = f"{opts['redimensionamento']}_{norm}{seg}"
    return OUT / name / f"features_{name}.csv"


def load_features(opts):
    path = features_path(opts)
    if path.exists():  # extração leva alguns minutos: reaproveita (apague o CSV para recalcular)
        print(f"Usando features já extraídas: {path}")
        return pd.read_csv(path)
    df, skipped = build_dataset(opts)
    if skipped:
        print(f"{len(skipped)} imagens ignoradas (corrompidas):", *skipped, sep="\n  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def show_image(ax, img, title=None, ylabel=None):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)


def plot_preprocessing_examples(opts, path, per_class=2):
    """Etapas do pré-processamento em células de exemplo (fundo descartado mostrado em branco)."""
    rng = np.random.default_rng(SEED)
    samples = []
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        samples += [(class_dir.name, files[i]) for i in rng.choice(len(files), per_class, replace=False)]

    titles = ["original", f"redimensionada ({opts['redimensionamento']})", "máscara (após erosão)",
              "célula segmentada", "cor normalizada"]
    fig, axes = plt.subplots(len(samples), len(titles), figsize=(2.2 * len(titles), 2.3 * len(samples)),
                             facecolor=SURFACE, squeeze=False)
    for r, (row, (cls, f)) in enumerate(zip(axes, samples)):
        full = Image.open(f)
        small = downscale(full, opts["redimensionamento"])
        rgb = np.asarray(small, dtype=np.float32) / 255
        mask = cell_mask(full, small.size)
        cell = mask[..., None]
        panels = [np.asarray(full, dtype=np.float32) / 255, rgb, mask.astype(np.float32),
                  np.where(cell, rgb, 1.0), np.where(cell, normalize_color(rgb, mask), 1.0)]
        for c, (ax, img) in enumerate(zip(row, panels)):
            show_image(ax, img, title=titles[c] if r == 0 else None, ylabel=cls if c == 0 else None)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_errors(test, pred, model_name, path, n=8):
    """Exemplos aleatórios de erros no teste: falsos negativos (parasitada prevista como não infectada)
    e falsos positivos (não infectada prevista como parasitada). Título de cada imagem: paciente."""
    rng = np.random.default_rng(SEED)
    cases = [("Parasitized", "Uninfected", "Falsos negativos: parasitada prevista como não infectada"),
             ("Uninfected", "Parasitized", "Falsos positivos: não infectada prevista como parasitada")]
    true = test["classe"].to_numpy()
    fig, axes = plt.subplots(len(cases), n, figsize=(1.7 * n, 4.8), facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, hspace=0.6, wspace=0.08)
    for row, (real, predicted, label) in zip(axes, cases):
        idx = np.flatnonzero((true == real) & (pred == predicted))
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else []
        for ax in row:
            ax.axis("off")
        for ax, i in zip(row, pick):
            ax.imshow(np.asarray(Image.open(DATA / real / test.at[i, "id"])))
            ax.set_title(test.at[i, "paciente"], fontsize=8, color=INK_2)
        row[0].text(0, 1.32, f"{label} ({fmt_int(len(idx))} no teste)", transform=row[0].transAxes,
                    fontsize=10, color=INK, ha="left", va="bottom")
    fig.suptitle(f"Exemplos de erros no conjunto de teste: {model_name}", color=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def evaluate(train, test, labels, folder, cols):
    """Busca de hiperparâmetros de cada modelo; salva métricas de validação cruzada e de teste em folder,
    as predições de teste por imagem e exemplos de erros do melhor modelo (maior F1 macro na CV)."""
    val_dir, test_dir = folder / "validacao_cruzada", folder / "testset"
    for d in (val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)
    X_train, y_train, g_train = train[cols], train["classe"], train["paciente"]
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_preds, test_preds, best = {}, {}, []
    predictions = test[["id", "classe", "paciente"]].copy()
    for slug, (name, estimator, grid) in MODELS.items():
        print(f"\n== {name} ==")
        search = tune(estimator, grid, X_train, y_train, g_train, cv)
        grid_table(search).to_csv(val_dir / f"grid_{slug}.csv", index=False)
        params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        best.append({"modelo": name, "f1_macro_cv": search.best_score_, **params})
        print(f"melhores hiperparâmetros: {params} | F1 macro CV: {search.best_score_:.4f}")

        # predições out-of-fold com os melhores hiperparâmetros (mesmos folds da busca)
        cv_preds[name] = cross_val_predict(search.best_estimator_, X_train, y_train,
                                           groups=g_train, cv=cv, n_jobs=-1)
        # best_estimator_ já foi reajustado em todo o treino (refit=True)
        test_preds[name] = search.predict(test[cols])
        predictions[f"predito_{slug}"] = test_preds[name]

    pd.DataFrame(best).to_csv(val_dir / "melhores_hiperparametros.csv", index=False)
    save_results(val_dir, cv_preds, y_train, labels, "Validação cruzada (treino, out-of-fold)",
                 "matrizes_confusao_cv.png")
    save_results(test_dir, test_preds, test["classe"], labels, "Conjunto de teste", "matrizes_confusao_teste.png")
    predictions.to_csv(test_dir / "predicoes.csv", index=False)
    top = max((b for b in best if b["modelo"] != MODELS["baseline"][0]), key=lambda b: b["f1_macro_cv"])
    plot_errors(test, test_preds[top["modelo"]], top["modelo"], test_dir / "erros_exemplos.png")


def compare():
    """F1 macro de validação cruzada e métricas de teste lado a lado, por configuração e modelo.

    Lê as métricas salvas em outputs/<config>/ de todas as configurações que já foram rodadas.
    """
    tables, opt_cols = [], list(next(iter(CONFIGS.values())))
    for config in CONFIGS:
        try:
            cv_overall = pd.read_csv(OUT / config / "validacao_cruzada" / "metricas_gerais.csv")
            test_overall = pd.read_csv(OUT / config / "testset" / "metricas_gerais.csv")
        except FileNotFoundError:  # configuração ainda não rodada
            continue
        cv_part = cv_overall[["modelo", "f1_macro"]].rename(columns={"f1_macro": "f1_macro_cv"})
        test_part = test_overall[["modelo", "acuracia", "f1_macro"]].rename(
            columns={"acuracia": "acuracia_teste", "f1_macro": "f1_macro_teste"})
        tables.append(cv_part.merge(test_part, on="modelo").assign(configuracao=config, **CONFIGS[config]))
    table = pd.concat(tables, ignore_index=True)
    table = table[["configuracao", *opt_cols, "modelo", "f1_macro_cv", "acuracia_teste", "f1_macro_teste"]]
    table.to_csv(OUT / "comparacao_preprocessamento.csv", index=False)
    print("\n== Comparação entre configurações de pré-processamento ==")
    print(table.round(4).to_string(index=False))


def main(selected):
    unknown = [c for c in selected if c not in CONFIGS]
    if unknown:
        sys.exit(f"Configuração desconhecida: {', '.join(unknown)}. Opções: {', '.join(CONFIGS)}")
    plot_preprocessing_examples(CONFIGS[FINAL], OUT / "exemplos_segmentacao.png")
    for i, config in enumerate(selected or CONFIGS):
        print(f"\n######## {config} ########")
        opts = CONFIGS[config]
        df = load_features(opts)
        train, test = split_train_test(df)  # mesmas imagens e semente: divisão idêntica nas configurações
        if i == 0:
            describe_split(train, test)
        labels = sorted(df["classe"].unique())
        evaluate(train, test, labels, OUT / config, FEATURE_SETS[opts["features"]])
    compare()


if __name__ == "__main__":
    main(sys.argv[1:])
