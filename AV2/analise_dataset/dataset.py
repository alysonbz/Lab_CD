import urllib.request
import zipfile
from .config import DATA_DIR, DATASET_DIR, DATASET_URL, CLASSES, IMAGE_EXTENSIONS

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"Dataset já encontrado em: {DATASET_DIR}")
        return
    archive = DATA_DIR / "dataset-resized.zip"
    print("Baixando dataset TrashNet...")
    urllib.request.urlretrieve(DATASET_URL, archive)
    print("Extraindo dataset...")
    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extractall(DATA_DIR)
    print(f"Dataset disponível em: {DATASET_DIR}")

def run():
    download_and_extract()
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_DIR}")
    print("\\nQuantidade de imagens por pasta:")
    for folder in sorted(DATASET_DIR.iterdir()):
        if folder.is_dir():
            count = sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"{folder.name}: {count}")
