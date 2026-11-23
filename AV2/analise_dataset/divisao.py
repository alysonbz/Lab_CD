import shutil
from sklearn.model_selection import train_test_split
from .config import CLEAN_DIR, SPLIT_DIR, CLASSES, IMAGE_EXTENSIONS

def run():
    paths, labels = [], []
    for class_name in CLASSES:
        folder = CLEAN_DIR / class_name
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(class_name)

    train_x, test_x, train_y, test_y = train_test_split(
        paths, labels, test_size=0.20, random_state=42, stratify=labels
    )
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for split in ("treinamento", "teste"):
        for class_name in CLASSES:
            (SPLIT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    for path, class_name in zip(train_x, train_y):
        shutil.copy2(path, SPLIT_DIR / "treinamento" / class_name / path.name)
    for path, class_name in zip(test_x, test_y):
        shutil.copy2(path, SPLIT_DIR / "teste" / class_name / path.name)

    print(f"\\nDivisão concluída: {len(train_x)} treino; {len(test_x)} teste.")
    for split in ("treinamento", "teste"):
        print(f"\\n{split.upper()}")
        for class_name in CLASSES:
            folder = SPLIT_DIR / split / class_name
            print(f"{class_name}: {sum(1 for p in folder.iterdir() if p.is_file())}")
