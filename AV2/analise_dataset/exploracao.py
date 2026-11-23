import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from .config import DATASET_DIR, OUTPUT_DIR, CLASSES, IMAGE_EXTENSIONS

def run():
    rows = []
    for class_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        rows.append({"Classe": class_dir.name, "Quantidade": count})
    df = pd.DataFrame(rows)
    total = df["Quantidade"].sum()
    df["Porcentagem (%)"] = (df["Quantidade"] / total * 100).round(2)
    print("\\nDistribuição das classes:")
    print(df.to_string(index=False))
    print("Total de imagens:", total)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["Classe"], df["Quantidade"])
    ax.set(title="Distribuição de imagens por classe", xlabel="Classes de resíduos",
           ylabel="Quantidade de imagens")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distribuicao_classes.png", dpi=150)
    plt.close(fig)

    rng = random.Random(42)
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(10, 18))
    for row, class_name in enumerate(CLASSES):
        folder = DATASET_DIR / class_name
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        samples = rng.sample(files, min(3, len(files)))
        for col in range(3):
            axes[row, col].axis("off")
            if col < len(samples):
                with Image.open(samples[col]) as img:
                    axes[row, col].imshow(img.copy())
            if col == 1:
                axes[row, col].set_title(class_name)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "amostras_dataset.png", dpi=150)
    plt.close(fig)
