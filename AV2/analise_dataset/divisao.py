
import json
import shutil

from sklearn.model_selection import train_test_split

from .config import (
    CLEAN_DIR,
    SPLIT_DIR,
    OUTPUT_DIR,
    CLASSES,
    IMAGE_EXTENSIONS,
    RANDOM_STATE
)


def run():
    print("\n" + "=" * 60)
    print("DIVISÃO DO DATASET")
    print("=" * 60)
    paths = []
    labels = []

    for class_name in CLASSES:
        folder = CLEAN_DIR / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Pasta da classe não encontrada: {folder}"
            )

        imagens = [
            path for path in sorted(folder.iterdir())
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(imagens) < 2:
            raise ValueError(
                f"A classe '{class_name}' possui menos de "
                "2 imagens. Não é possível realizar a divisão "
                "estratificada."
            )
        
        paths.extend(imagens)
        labels.extend([class_name] * len(imagens))

    if not paths:
        raise ValueError("Nenhuma imagem foi encontrada no dataset.")

    train_x, test_x, train_y, test_y = train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    for split in ("treinamento", "teste"):
        for class_name in CLASSES:
            pasta = SPLIT_DIR / split / class_name
            pasta.mkdir(parents=True, exist_ok=True)

    registros_treino = []

    for path, class_name in zip(train_x, train_y):
        destino = (
            SPLIT_DIR / "treinamento" / class_name / path.name
        )
        shutil.copy2(path, destino)
        registros_treino.append({
            "path": str(destino.resolve()),
            "label": class_name
        })

    registros_teste = []
    for path, class_name in zip(test_x, test_y):
        destino = (
            SPLIT_DIR / "teste" / class_name / path.name
        )
        shutil.copy2(path, destino)
        registros_teste.append({
            "path": str(destino.resolve()),
            "label": class_name
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dados = {
        "train": registros_treino,
        "test": registros_teste
    }
    split_path = OUTPUT_DIR / "split.json"
    with open(split_path, "w", encoding="utf-8") as arquivo:
        json.dump(
            dados,
            arquivo,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nDivisão concluída: "
        f"{len(train_x)} imagens para treino; "
        f"{len(test_x)} imagens para teste."
    )
    for split in ("treinamento", "teste"):
        print(f"\n{split.upper()}")
        for class_name in CLASSES:
            folder = SPLIT_DIR / split / class_name
            quantidade = sum(
                1 for p in folder.iterdir() if p.is_file()
            )
            print(f"{class_name}: {quantidade}")

    print(f"\nArquivo de divisão salvo em: {split_path}")
    print("\nDivisão do dataset concluída!")


if __name__ == "__main__":
    run()
