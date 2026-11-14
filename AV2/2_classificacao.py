"""Classificação do dataset cell_images com SVM e Regressão Logística.

1. extrai características de cor e textura de cada imagem (dataframe em outputs/features.csv)
2. divide em treino/teste por paciente (StratifiedGroupKFold), sem paciente nos dois conjuntos
3. ajusta hiperparâmetros por GridSearchCV com validação cruzada agrupada por paciente,
   somente no treino; o StandardScaler fica dentro do pipeline (ajustado só no treino de cada fold)
4. métricas e matrizes de confusão das predições out-of-fold -> outputs/validacao_cruzada/
5. avaliação final no conjunto de teste separado no passo 2 -> outputs/testset/
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern
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
VAL_DIR = OUT / "validacao_cruzada"
TEST_DIR = OUT / "testset"
FEATURES_CSV = OUT / "features.csv"
for folder in (OUT, VAL_DIR, TEST_DIR):
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEED = 42
N_FOLDS = 5  # 1/5 dos pacientes vai para teste; mesmo nº de folds na validação cruzada
SIZE = 64  # imagens têm tamanhos variados: redimensiona para SIZE x SIZE
BG_THRESHOLD = 0.04  # fundo é preto: pixels com cinza abaixo disso ficam fora das estatísticas
HIST_BINS = 16
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2  # padrões uniformes + "não uniforme"

FEATURE_NAMES = (
    [f"{space}_{stat}_{c}" for space, chans in (("rgb", "RGB"), ("hsv", "HSV"))
     for stat in ("media", "desvio") for c in chans]
    + [f"hist_{c}_{i}" for c in "RGB" for i in range(HIST_BINS)]
    + [f"lbp_{i}" for i in range(LBP_BINS)]
)

# intervalos de teste dos hiperparâmetros (prefixo clf__ = passo do pipeline)
MODELS = {
    "svm": (
        "SVM (RBF)",
        SVC(kernel="rbf", cache_size=1000),
        {"clf__C": np.logspace(-1, 2, 4), "clf__gamma": np.logspace(-3, -1, 3)},
    ),
    "regressao_logistica": (
        "Regressão Logística",
        LogisticRegression(max_iter=5000),
        {"clf__C": np.logspace(-4, 2, 7)},
    ),
}

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def fmt_pct(p):
    return f"{p:.1%}".replace(".", ",")


def patient_id(filename):
    """C<n> no início do nome identifica o paciente/lâmina (ex.: C33P1thinF_IMG_... -> C33)."""
    match = re.match(r"C\d+", filename)
    return match.group() if match else filename.split("_IMG")[0]


def extract(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255
    gray = rgb2gray(rgb)
    mask = gray > BG_THRESHOLD
    if mask.sum() < 10:  # imagem quase toda escura: usa tudo
        mask = np.ones_like(mask)
    n = mask.sum()

    feats = []
    for channels in (rgb, rgb2hsv(rgb)):
        px = channels[mask]
        feats += [px.mean(axis=0), px.std(axis=0)]
    for c in range(3):
        hist, _ = np.histogram(rgb[..., c][mask], bins=HIST_BINS, range=(0, 1))
        feats.append(hist / n)
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=LBP_BINS, range=(0, LBP_BINS))
    feats.append(hist / n)
    return np.concatenate(feats)


def build_dataset():
    rows, feats, skipped = [], [], []
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for f in tqdm(files, desc=f"features {class_dir.name}", unit="img"):
            try:
                feats.append(extract(f))
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
    split[["id", "classe", "paciente", "conjunto"]].to_csv(OUT / "divisao_treino_teste.csv", index=False)
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
    vmax = max(m.max() for m in mats.values())  # mesma escala nas duas matrizes
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


def save_results(folder, preds, y_true, labels, title):
    results = [compute_metrics(name, y_true, y_pred, labels) for name, y_pred in preds.items()]
    per_class = pd.concat([pc for pc, _ in results], ignore_index=True)
    overall = pd.DataFrame([ov for _, ov in results])
    per_class.to_csv(folder / "metricas_por_classe.csv", index=False)
    overall.to_csv(folder / "metricas_gerais.csv", index=False)
    plot_confusion(preds, y_true, labels, title, folder / "matrizes_confusao.png")
    print(f"\n-- {title} --")
    print(per_class.round(4).to_string(index=False))
    print(overall.round(4).to_string(index=False))


def main():
    if FEATURES_CSV.exists():  # extração leva vários minutos: reaproveita (apague o CSV para recalcular)
        print(f"Usando features já extraídas: {FEATURES_CSV}")
        df = pd.read_csv(FEATURES_CSV)
    else:
        df, skipped = build_dataset()
        if skipped:
            print(f"{len(skipped)} imagens ignoradas (corrompidas):", *skipped, sep="\n  ")
        df.to_csv(FEATURES_CSV, index=False)

    train, test = split_train_test(df)
    describe_split(train, test)
    labels = sorted(df["classe"].unique())
    X_train, y_train, g_train = train[FEATURE_NAMES], train["classe"], train["paciente"]
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_preds, test_preds, best = {}, {}, []
    for slug, (name, estimator, grid) in MODELS.items():
        print(f"\n== {name} ==")
        search = tune(estimator, grid, X_train, y_train, g_train, cv)
        grid_table(search).to_csv(VAL_DIR / f"grid_{slug}.csv", index=False)
        params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        best.append({"modelo": name, "f1_macro_cv": search.best_score_, **params})
        print(f"melhores hiperparâmetros: {params} | F1 macro CV: {search.best_score_:.4f}")

        # predições out-of-fold com os melhores hiperparâmetros (mesmos folds da busca)
        cv_preds[name] = cross_val_predict(search.best_estimator_, X_train, y_train,
                                           groups=g_train, cv=cv, n_jobs=-1)
        # best_estimator_ já foi reajustado em todo o treino (refit=True)
        test_preds[name] = search.predict(test[FEATURE_NAMES])

    pd.DataFrame(best).to_csv(VAL_DIR / "melhores_hiperparametros.csv", index=False)
    save_results(VAL_DIR, cv_preds, y_train, labels, "Validação cruzada (treino, out-of-fold)")
    save_results(TEST_DIR, test_preds, test["classe"], labels, "Conjunto de teste")


if __name__ == "__main__":
    main()
