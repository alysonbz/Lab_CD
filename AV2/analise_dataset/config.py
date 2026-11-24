from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
DATASET_DIR = DATA_DIR / "dataset-resized"
CLEAN_DIR = DATA_DIR / "dataset_limpo"
SPLIT_DIR = DATA_DIR / "dataset_dividido"

DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42

# Pares identificados na análise manual do notebook como duplicatas.
DUPLICATES_TO_EXCLUDE = {
    "glass/glass115.jpg",
    "metal/metal91.jpg",
    "glass/glass389.jpg",
    "plastic/plastic332.jpg",
    "glass/glass176.jpg",
    "plastic/plastic152.jpg",
}

def prepare_directories():
    for path in (DATA_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
