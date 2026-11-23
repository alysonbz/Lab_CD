import hashlib
import shutil
from PIL import Image
from .config import DATASET_DIR, CLEAN_DIR, CLASSES, IMAGE_EXTENSIONS, DUPLICATES_TO_EXCLUDE

def run():
    corrupted, hashes, duplicates = [], {}, []
    for class_name in CLASSES:
        folder = DATASET_DIR / class_name
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                corrupted.append(str(path))
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in hashes:
                duplicates.append((hashes[digest], str(path)))
            else:
                hashes[digest] = str(path)

    print(f"\\nImagens corrompidas: {len(corrupted)}")
    print(f"Duplicatas exatas encontradas: {len(duplicates)}")
    for item in corrupted:
        print("Corrompida:", item)
    for first, second in duplicates:
        print("Duplicata:", first, "<->", second)

    # Recria a pasta limpa para evitar resíduos de execuções anteriores.
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    for class_name in CLASSES:
        source = DATASET_DIR / class_name
        target = CLEAN_DIR / class_name
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            relative = f"{class_name}/{path.name}"
            if path.suffix.lower() in IMAGE_EXTENSIONS and relative not in DUPLICATES_TO_EXCLUDE:
                shutil.copy2(path, target / path.name)

    print("\\nDataset limpo criado em:", CLEAN_DIR)
    total = 0
    for class_name in CLASSES:
        count = sum(1 for f in (CLEAN_DIR / class_name).iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS)
        total += count
        print(f"{class_name}: {count} imagens")
    print("Total após limpeza:", total)
