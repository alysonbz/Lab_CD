"""
PASSO 5 / TÓPICO 5 - Treinamento e avaliação dos classificadores.

Algoritmos : KNN, Random Forest, SVM (kernel RBF) + DummyClassifier (classe mais frequente).
Variantes  : 'inteira' (cor/textura/gradientes na imagem toda) e 'segmentada' (restrita à máscara).
Configurações de características: C1 (cor), C2 (cor+textura), C3 (+gradientes), C4 (+forma, só segmentada).

Metodologia (sem vazamento):
  * hiperparâmetros escolhidos por GridSearchCV com StratifiedGroupKFold(3) em uma subamostra
    do TREINO (mesma subamostra para todas as tarefas => condições equivalentes), métrica F1 macro;
  * StandardScaler dentro do Pipeline (ajustado só nos dados de treino de cada dobra / do treino final);
  * modelo final reajustado em TODO o treino com os melhores hiperparâmetros;
  * o TESTE só é usado depois, para a avaliação final.

Cada tarefa é salva em cache (retomável).  Uso:  python step05_train_evaluate.py [--max-seconds N] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import features as F
from step02_preprocessing import train_sample, cv_scores
from utils import log, timer, savefig, save_table, require, pretty_class, split_class

RES_DIR = config.DIR_CACHE / "resultados"
RES_DIR.mkdir(exist_ok=True)
RAPIDO = False          # modo rápido (ver --rapido); é (re)definido em cada processo por set_mode()


def set_mode(rapido: bool):
    """Define o modo no processo atual (os trabalhadores do joblib são processos separados)."""
    global RAPIDO, RES_DIR
    RAPIDO = bool(rapido)
    RES_DIR = config.DIR_CACHE / ("resultados_rapido" if RAPIDO else "resultados")
    RES_DIR.mkdir(exist_ok=True)


def cv_subsample_size():
    return 4000 if RAPIDO else config.CV_SUBSAMPLE

VARIANT_FILES = {"inteira": config.FEATURES_FULL_NPZ, "segmentada": config.FEATURES_SEG_NPZ}


def make_pipeline_and_grid(model: str):
    if model == "KNN":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=1))])
        grid = {"clf__n_neighbors": [3, 7, 15], "clf__weights": ["uniform", "distance"]}
    elif model == "RandomForest":
        # modo completo: 200 árvores, max_features ∈ {sqrt, 0,3};  modo rápido: 100 árvores, max_features ∈ {sqrt, 0,1}
        n_est = 100 if RAPIDO else 200
        pipe = Pipeline([("clf", RandomForestClassifier(n_estimators=n_est, n_jobs=1, random_state=config.SEED))])
        grid = {"clf__max_features": ["sqrt", 0.1 if RAPIDO else 0.3],
                "clf__class_weight": [None, "balanced_subsample"]}
    elif model == "SVM":
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", gamma="scale", cache_size=1000, random_state=config.SEED))])
        grid = {"clf__C": [1, 10, 100], "clf__class_weight": [None, "balanced"]}
    else:
        raise ValueError(model)
    return pipe, grid


def load_variant(variant):
    d = np.load(VARIANT_FILES[variant])
    return d["X"], list(d["names"])


def load_split():
    sp = pd.read_csv(config.SPLIT_CSV)
    return sp, (sp["split"] == "train").to_numpy(), (sp["split"] == "test").to_numpy()


def task_key(variant, cfg, model):
    return f"{variant}__{cfg}__{model}"


def run_task(variant: str, cfg: str, model: str, force: bool = False, rapido: bool = False):
    set_mode(rapido)
    key = task_key(variant, cfg, model)
    out = RES_DIR / f"{key}.json"
    if out.exists() and not force:
        return key
    t0 = time.time()
    X, names = load_variant(variant)
    sp, tr, te = load_split()
    require(len(sp) == len(X), "Features e split desalinhados: rode os Passos 1 e 4.")
    cols = F.columns_for(names, config.FEATURE_CONFIGS[cfg])
    require(len(cols) > 0, f"Nenhuma coluna para {cfg} em {variant}.")
    Xtr, Xte = X[tr][:, cols], X[te][:, cols]
    ytr, yte = sp.loc[tr, "class"].to_numpy(), sp.loc[te, "class"].to_numpy()
    gtr = sp.loc[tr, "group_id"].to_numpy()

    # subamostra de TREINO (mesma para todas as tarefas) para a busca de hiperparâmetros
    sub = train_sample(cv_subsample_size())
    trdf = sp[tr].reset_index(drop=True)
    pos = trdf.reset_index().set_index("image_id").loc[sub["image_id"], "index"].to_numpy()
    pipe, grid = make_pipeline_and_grid(model)
    sg = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=sg, n_jobs=1, refit=False)
    gs.fit(Xtr[pos], ytr[pos], groups=gtr[pos])
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
    cv_f1 = float(gs.best_score_)
    t_search = time.time() - t0

    pipe.set_params(**best)
    t1 = time.time()
    pipe.fit(Xtr, ytr)
    t_fit = time.time() - t1
    t2 = time.time()
    pred = pipe.predict(Xte)
    t_pred = time.time() - t2

    classes = np.array(sorted(np.unique(ytr)))
    p, r, f, s = precision_recall_fscore_support(yte, pred, labels=classes, zero_division=0)
    np.save(RES_DIR / f"{key}__pred.npy", pred.astype(str))
    np.save(RES_DIR / f"{key}__cm.npy", confusion_matrix(yte, pred, labels=classes))
    pd.DataFrame({"classe": classes, "precisao": p, "recall": r, "f1": f, "suporte": s}).to_csv(
        RES_DIR / f"{key}__por_classe.csv", index=False)
    joblib.dump({"pipeline": pipe, "columns": [names[i] for i in cols], "variant": variant, "config": cfg},
                RES_DIR / f"{key}__modelo.joblib", compress=3)
    res = {"variante": variant, "config": cfg, "modelo": model, "n_features": len(cols),
           "melhores_hiperparametros": best, "f1_macro_cv_treino": cv_f1,
           "acuracia_teste": float(accuracy_score(yte, pred)),
           "f1_macro_teste": float(f1_score(yte, pred, average="macro")),
           "f1_ponderado_teste": float(f1_score(yte, pred, average="weighted")),
           "precisao_macro_teste": float(p.mean()), "recall_macro_teste": float(r.mean()),
           "segundos_busca": t_search, "segundos_treino_final": t_fit, "segundos_predicao": t_pred,
           "modo": "rapido" if RAPIDO else "completo"}
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return key


def run_dummy():
    sp, tr, te = load_split()
    ytr, yte = sp.loc[tr, "class"].to_numpy(), sp.loc[te, "class"].to_numpy()
    rows = []
    for strat in ("most_frequent", "stratified"):
        d = DummyClassifier(strategy=strat, random_state=config.SEED).fit(np.zeros((len(ytr), 1)), ytr)
        p = d.predict(np.zeros((len(yte), 1)))
        rows.append({"referencia": f"DummyClassifier({strat})", "acuracia_teste": accuracy_score(yte, p),
                     "f1_macro_teste": f1_score(yte, p, average="macro", zero_division=0),
                     "f1_ponderado_teste": f1_score(yte, p, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def all_tasks():
    tasks = []
    for variant, cfgs in (("inteira", ["C1_COR", "C2_COR+TEXTURA", "C3_COR+TEXTURA+GRADIENTES"]),
                          ("segmentada", list(config.FEATURE_CONFIGS))):
        for cfg, m in product(cfgs, config.MODELS):
            tasks.append((variant, cfg, m))
    return tasks


def _task_weight(t):
    """Custo relativo estimado (maior primeiro => melhor balanceamento entre núcleos)."""
    variant, cfg, model = t
    dims = {"C1": 1, "C2": 1.6, "C3": 7, "C4": 7}[cfg[:2]]
    w = {"RandomForest": 10, "SVM": 3, "KNN": 0.2}[model]
    return w * dims


def _log_done(v, c, m):
    r = json.loads((RES_DIR / f"{task_key(v, c, m)}.json").read_text(encoding="utf-8"))
    log(f"   {v:10s} {c:27s} {m:12s} CV-F1={r['f1_macro_cv_treino']:.3f} | teste: acc={r['acuracia_teste']:.3f} "
        f"F1macro={r['f1_macro_teste']:.3f} | busca {r['segundos_busca']:.0f}s treino {r['segundos_treino_final']:.0f}s")


def run_all_tasks(max_seconds=None, force=False, rapido=False, modelos=None):
    t0 = time.time()
    tasks = [t for t in all_tasks() if (modelos is None or t[2] in modelos)]
    pending = [t for t in tasks if force or not (RES_DIR / f"{task_key(*t)}.json").exists()]
    pending.sort(key=_task_weight, reverse=True)
    log(f"   modo={'RÁPIDO' if rapido else 'completo'} | {len(tasks) - len(pending)} tarefas em cache | "
        f"{len(pending)} pendentes | tarefas simultâneas={config.TASK_JOBS}")
    if not pending:
        return True
    if max_seconds:                                  # execução em lotes interrompíveis (útil p/ depurar)
        step = config.TASK_JOBS
        for i in range(0, len(pending), step):
            if time.time() - t0 > max_seconds:
                log(f"   limite de tempo atingido; rode de novo para continuar ({len(pending) - i} pendentes)")
                return False
            batch = pending[i:i + step]
            Parallel(n_jobs=min(step, len(batch)))(delayed(run_task)(v, c, m, force, rapido) for v, c, m in batch)
            for v, c, m in batch:
                _log_done(v, c, m)
        return True
    jobs = (delayed(run_task)(v, c, m, force, rapido) for v, c, m in pending)
    n_jobs = min(config.TASK_JOBS, len(pending))
    try:                                             # joblib >= 1.3: registra cada tarefa assim que termina
        it = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs)
        for key in it:
            v, c, m = key.split("__")
            _log_done(v, c, m)
    except TypeError:
        Parallel(n_jobs=n_jobs)(jobs)
        for v, c, m in pending:
            _log_done(v, c, m)
    return True


# =========================================================================== #
# AGREGAÇÃO, GRÁFICOS E ANÁLISE DE ERROS
# =========================================================================== #
def collect_results():
    rows = []
    for t in all_tasks():
        f = RES_DIR / f"{task_key(*t)}.json"
        if not f.exists():
            log(f"   AVISO: tarefa sem resultado (ignorada): {task_key(*t)}")
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        r["melhores_hiperparametros"] = json.dumps(r["melhores_hiperparametros"])
        rows.append(r)
    return pd.DataFrame(rows)


def plot_comparison(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, var in zip(axes, ("inteira", "segmentada")):
        d = res[res["variante"] == var]
        sns.barplot(data=d, x="config", y="f1_macro_teste", hue="modelo", ax=ax)
        ax.set_title(f"Imagem {var}: F1 macro no TESTE")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=6)
    savefig(fig, "05_comparacao_configuracoes_modelos.png")


def plot_cm(key, title, fname):
    cm = np.load(RES_DIR / f"{key}__cm.npy")
    classes = pd.read_csv(RES_DIR / f"{key}__por_classe.csv")["classe"].tolist()
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cmn, ax=ax, cmap="Blues", vmin=0, vmax=1, xticklabels=[pretty_class(c)[:26] for c in classes],
                yticklabels=[pretty_class(c)[:26] for c in classes], cbar_kws={"label": "proporção da classe verdadeira"})
    ax.tick_params(labelsize=6)
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe verdadeira")
    ax.set_title(title)
    savefig(fig, fname)


def error_analysis(key, sp_test):
    pred = np.load(RES_DIR / f"{key}__pred.npy")
    y = sp_test["class"].to_numpy()
    classes = pd.read_csv(RES_DIR / f"{key}__por_classe.csv")
    err = pred != y
    pairs = (pd.DataFrame({"verdadeira": y[err], "prevista": pred[err]}).value_counts()
             .rename("n_erros").reset_index().head(20))
    n_true = pd.Series(y).value_counts()
    pairs["pct_da_classe_verdadeira"] = [100 * n / n_true[t] for t, n in zip(pairs["verdadeira"], pairs["n_erros"])]
    pairs["mesma_cultura"] = [split_class(a)[0] == split_class(b)[0] for a, b in zip(pairs["verdadeira"], pairs["prevista"])]
    same = np.mean([split_class(a)[0] == split_class(b)[0] for a, b in zip(y[err], pred[err])])
    # amostra de erros p/ galeria
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(np.where(err)[0], size=min(24, int(err.sum())), replace=False)
    fig, axes = plt.subplots(4, 6, figsize=(13, 9.4))
    for ax, i in zip(axes.ravel(), idx):
        im = cv2.imread(str(config.RAW_COLOR_DIR / sp_test["image_id"].iloc[i]))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"V: {pretty_class(y[i])[:24]}\nP: {pretty_class(pred[i])[:24]}", fontsize=6.5)
        ax.axis("off")
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle("Exemplos de imagens classificadas incorretamente (V=verdadeira, P=prevista)")
    savefig(fig, "05_erros_exemplos.png", dpi=110)
    return pairs, float(same), classes.sort_values("f1")


def ablation(force=False):
    """Contribuição isolada de cada grupo (RF, CV agrupada na subamostra de TREINO)."""
    out = config.DIR_CACHE / ("ablacao_rapido.csv" if RAPIDO else "ablacao.csv")
    if out.exists() and not force:
        return pd.read_csv(out)
    sp, tr, _ = load_split()
    sub = train_sample(cv_subsample_size())
    trdf = sp[tr].reset_index(drop=True)
    pos = trdf.reset_index().set_index("image_id").loc[sub["image_id"], "index"].to_numpy()
    y, g = trdf["class"].to_numpy()[pos], trdf["group_id"].to_numpy()[pos]
    rows = []
    for grp, pref in config.ABLATION_GROUPS.items():
        var = "segmentada" if grp == "FORMA" else "inteira"
        X, names = load_variant(var)
        cols = F.columns_for(names, pref)
        Xs = X[tr][:, cols][pos]
        f1, acc = cv_scores(Xs, y, g, lambda: RandomForestClassifier(100, n_jobs=config.N_JOBS, random_state=config.SEED))
        rows.append({"grupo": grp, "n_features": len(cols), "f1_macro_cv": f1.mean(), "dp": f1.std(), "acuracia_cv": acc.mean()})
        log(f"   ablação {grp:20s} n={len(cols):3d} F1macro(CV)={f1.mean():.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def finalize():
    res = collect_results()
    save_table(res, "05_resultados_todas_as_tarefas.csv")
    tempos = res[["variante", "config", "modelo", "segundos_busca", "segundos_treino_final", "segundos_predicao"]].copy()
    tempos["segundos_total"] = tempos[["segundos_busca", "segundos_treino_final", "segundos_predicao"]].sum(axis=1)
    save_table(tempos.sort_values("segundos_total", ascending=False), "05_tempos_por_tarefa.csv")
    dummy = run_dummy()
    save_table(dummy, "05_baseline_dummy.csv")
    log("\n" + dummy.round(4).to_string(index=False))
    plot_comparison(res)
    best_cv = res.sort_values("f1_macro_cv_treino", ascending=False).iloc[0]
    best_test = res.sort_values("f1_macro_teste", ascending=False).iloc[0]
    key = task_key(best_cv["variante"], best_cv["config"], best_cv["modelo"])
    cv_, cfg_, mod_ = best_cv["variante"], best_cv["config"], best_cv["modelo"]
    log(f"   modelo escolhido por VALIDAÇÃO (treino): {key} (CV-F1={best_cv['f1_macro_cv_treino']:.3f}; "
        f"teste F1macro={best_cv['f1_macro_teste']:.3f})")
    sp, tr, te = load_split()
    sp_test = sp[te].reset_index(drop=True)
    plot_cm(key, f"Matriz de confusão normalizada — {key}", "05_matriz_confusao_modelo_escolhido.png")
    # matriz de confusão do melhor de cada modelo (por CV) para comparação
    for m in config.MODELS:
        r = res[res["modelo"] == m].sort_values("f1_macro_cv_treino", ascending=False).iloc[0]
        plot_cm(task_key(r["variante"], r["config"], m), f"{m}: {r['variante']} / {r['config']}",
                f"05_matriz_confusao_{m}.png")
    # relatório de classificação do modelo escolhido
    pred = np.load(RES_DIR / f"{key}__pred.npy")
    rep = classification_report(sp_test["class"], pred, digits=3, zero_division=0)
    (config.DIR_REPORT / "05_classification_report_modelo_escolhido.txt").write_text(f"{key}\n\n{rep}", encoding="utf-8")
    pc = pd.read_csv(RES_DIR / f"{key}__por_classe.csv")
    save_table(pc, "05_metricas_por_classe_modelo_escolhido.csv")
    pairs, same_crop, worst = error_analysis(key, sp_test)
    save_table(pairs, "05_pares_mais_confundidos.csv")
    save_table(worst, "05_classes_ordenadas_por_f1.csv")
    abl = ablation()
    save_table(abl, "05_ablacao_grupos.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    a = abl.sort_values("f1_macro_cv")
    ax.barh(a["grupo"], a["f1_macro_cv"], xerr=a["dp"], color="#4c72b0")
    ax.set_xlabel("F1 macro (CV agrupada, subamostra do treino; RF)")
    ax.set_title("Contribuição isolada de cada grupo de características")
    savefig(fig, "05_ablacao_grupos.png")
    # modelo final: guarda só o escolhido + remove joblibs dos demais (economiza disco)
    dst = config.DIR_MODELS / f"modelo_final_{key}.joblib"
    import shutil
    src = RES_DIR / f"{key}__modelo.joblib"
    if not dst.exists():
        if not src.exists():                       # execução retomada: modelo já descartado -> reajusta só este
            log("   reajustando o modelo escolhido (arquivo do modelo não está em cache)...")
            run_task(cv_, cfg_, mod_, force=True)
        shutil.copy(src, dst)
    for f in RES_DIR.glob("*__modelo.joblib"):
        f.unlink()
    summary = {"modelo_escolhido_por_cv": key, "cv_f1": float(best_cv["f1_macro_cv_treino"]),
               "teste_acuracia": float(best_cv["acuracia_teste"]), "teste_f1_macro": float(best_cv["f1_macro_teste"]),
               "melhor_por_teste": task_key(best_test["variante"], best_test["config"], best_test["modelo"]),
               "melhor_por_teste_f1": float(best_test["f1_macro_teste"]),
               "fracao_erros_mesma_cultura": same_crop, "arquivo_modelo": str(dst.name),
               "modo": "rapido" if RAPIDO else "completo", "cv_subamostra": cv_subsample_size(),
               "grades": {m: {k.replace("clf__", ""): v for k, v in make_pipeline_and_grid(m)[1].items()}
                          for m in config.MODELS},
               "rf_n_estimators": 100 if RAPIDO else 200}
    (config.DIR_REPORT / "resumo_passo5.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("PASSO 5 concluído.")


def main(max_seconds=None, force=False, rapido=False, modelos=None):
    set_mode(rapido)
    with timer("Treinamento/avaliação das tarefas"):
        done = run_all_tasks(max_seconds, force, rapido, modelos)
    if not done or modelos:
        return bool(done)
    with timer("Agregação, matrizes de confusão, análise de erros e ablação"):
        finalize()
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rapido", action="store_true",
                    help="modo rápido: RF com 100 árvores e max_features {sqrt,0.1}, subamostra de 4000 (resultados em cache separado)")
    ap.add_argument("--modelos", default=None, help="restringe a modelos (ex.: KNN,SVM); pula a etapa final")
    a = ap.parse_args()
    main(a.max_seconds, a.force, a.rapido, a.modelos.split(",") if a.modelos else None)
