"""
config.py - Configuração central do projeto AV2 (PlantVillage).

Todos os caminhos, parâmetros e sementes ficam aqui. Nenhum outro módulo
define caminhos "soltos": isso garante reprodutibilidade e consistência.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent

# Pasta onde o repositório oficial será clonado (pode ser sobrescrita pela
# variável de ambiente PV_REPO_DIR, útil se você já clonou o dataset).
REPO_DIR = Path(os.environ.get("PV_REPO_DIR", PROJECT_DIR / "data" / "PlantVillage-Dataset"))
REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
RAW_COLOR_DIR = REPO_DIR / "raw" / "color"
LEAF_MAP_JSON = REPO_DIR / "leaf_grouping" / "leaf-map.json"

OUT_DIR = Path(os.environ.get("AV2_OUT_DIR", PROJECT_DIR / "outputs"))
DIR_DATA = OUT_DIR / "dados"            # inventário, split, features (CSV)
DIR_FIG = OUT_DIR / "graficos"
DIR_TAB = OUT_DIR / "tabelas"
DIR_MODELS = OUT_DIR / "modelos"
DIR_REPORT = OUT_DIR / "relatorios"
DIR_CACHE = OUT_DIR / "cache"           # resultados intermediários (retomada)

for _d in (DIR_DATA, DIR_FIG, DIR_TAB, DIR_MODELS, DIR_REPORT, DIR_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

INVENTORY_CSV = DIR_DATA / "inventario_imagens.csv"      # 1 linha/imagem (+md5, grupo, split)
SPLIT_CSV = DIR_DATA / "split_treino_teste.csv"
PREPROC_DECISION_JSON = DIR_DATA / "decisao_preprocessamento.json"
SEG_DECISION_JSON = DIR_DATA / "decisao_segmentacao.json"
FEATURES_FULL_CSV = DIR_DATA / "features_imagem_completa.csv"
FEATURES_SEG_CSV = DIR_DATA / "features_imagem_segmentada.csv"
FEATURES_FULL_NPZ = DIR_CACHE / "features_full.npz"
FEATURES_SEG_NPZ = DIR_CACHE / "features_seg.npz"

# --------------------------------------------------------------------------- #
# Reprodutibilidade e desempenho
# --------------------------------------------------------------------------- #
SEED = 42
# processos paralelos (sobrescreva com a variável de ambiente AV2_N_JOBS se faltar RAM)
N_JOBS = int(os.environ.get("AV2_N_JOBS", max(1, (os.cpu_count() or 1))))
# tarefas de classificação simultâneas no Passo 5 (cada uma usa ~0,5-1 GB de RAM)
TASK_JOBS = max(1, min(N_JOBS, int(os.environ.get("AV2_TASK_JOBS", 4))))
CHUNK_SIZE = 2000                          # imagens por lote (retomada incremental)

# --------------------------------------------------------------------------- #
# Divisão treino/teste
# --------------------------------------------------------------------------- #
TEST_FRACTION = 0.20                       # ~20% => StratifiedGroupKFold com 5 dobras
N_SPLIT_FOLDS = 5

# --------------------------------------------------------------------------- #
# Pré-processamento / extração
# --------------------------------------------------------------------------- #
DEFAULT_IMG_SIZE = 128                     # decisão final é lida de PREPROC_DECISION_JSON
PREPROC_SUBSAMPLE = 3000                   # imagens (somente TREINO) p/ comparar pré-processamentos
SEG_REF_SAMPLE = 150                       # imagens p/ avaliar a segmentação contra a referência
SEG_FACTOR_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
SEG_DEFAULT_FACTOR = 0.8

# --------------------------------------------------------------------------- #
# Classificação
# --------------------------------------------------------------------------- #
CV_FOLDS = 3                               # validação cruzada (agrupada) no TREINO
CV_SUBSAMPLE = 6000                        # tamanho do subconjunto de TREINO p/ busca de hiperparâmetros
MODELS = ("KNN", "RandomForest", "SVM")    # + DummyClassifier como referência
# Configurações de características: nome -> prefixos de colunas
FEATURE_CONFIGS = {
    "C1_COR": ("rgb_", "hsv_", "lab_"),
    "C2_COR+TEXTURA": ("rgb_", "hsv_", "lab_", "lbp_", "glcm_"),
    "C3_COR+TEXTURA+GRADIENTES": ("rgb_", "hsv_", "lab_", "lbp_", "glcm_", "hog_", "edge_"),
    "C4_C3+FORMA": ("rgb_", "hsv_", "lab_", "lbp_", "glcm_", "hog_", "edge_", "shp_"),  # só imagem segmentada
}
# Grupos isolados para a análise de ablação
ABLATION_GROUPS = {
    "RGB": ("rgb_",), "HSV": ("hsv_",), "LAB": ("lab_",),
    "LBP": ("lbp_",), "GLCM": ("glcm_",), "HOG": ("hog_",),
    "BORDAS/GRADIENTES": ("edge_",), "FORMA": ("shp_",),
}
