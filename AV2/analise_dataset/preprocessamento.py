import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .config import CLEAN_DIR, OUTPUT_DIR, CLASSES, IMAGE_EXTENSIONS

def _first_image(class_name):
    folder = CLEAN_DIR / class_name
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"Nenhuma imagem encontrada para a classe {class_name}")
    return files[0]

def run():
    dimensions = {}
    for class_name in CLASSES:
        folder = CLEAN_DIR / class_name
        for path in folder.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(path) as image:
                    dimensions[image.size] = dimensions.get(image.size, 0) + 1
    print("\\nDimensões encontradas:")
    for size, count in sorted(dimensions.items()):
        print(f"{size}: {count} imagens")

    # RGB e canais Saturação/Brilho do HSV
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(12, 20))
    for row, class_name in enumerate(CLASSES):
        image = np.array(Image.open(_first_image(class_name)).convert("RGB"))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        axes[row, 0].imshow(image); axes[row, 0].set_title(f"{class_name} - RGB")
        axes[row, 1].imshow(hsv[:, :, 1], cmap="gray"); axes[row, 1].set_title("Saturação (HSV)")
        axes[row, 2].imshow(hsv[:, :, 2], cmap="gray"); axes[row, 2].set_title("Brilho (HSV)")
        for ax in axes[row]: ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "espaco_cores_hsv.png", dpi=150)
    plt.close(fig)

    # Filtros Gaussiano e de mediana
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(12, 20))
    for row, class_name in enumerate(CLASSES):
        image = np.array(Image.open(_first_image(class_name)).convert("RGB"))
        gaussian = cv2.GaussianBlur(image, (5, 5), 0)
        median = cv2.medianBlur(image, 5)
        for ax, data, title in zip(axes[row], (image, gaussian, median),
                                   (f"{class_name} - Original", "Filtro Gaussiano", "Filtro Mediana")):
            ax.imshow(data); ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reducao_ruido.png", dpi=150)
    plt.close(fig)

    # Normalização dos pixels
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(10, 18))
    last_original = last_normalized = None
    for row, class_name in enumerate(CLASSES):
        image = np.array(Image.open(_first_image(class_name)).convert("RGB"))
        normalized = image / 255.0
        axes[row, 0].imshow(image); axes[row, 0].set_title(f"{class_name} - Original")
        axes[row, 1].imshow(normalized); axes[row, 1].set_title(f"{class_name} - Normalizada")
        for ax in axes[row]: ax.axis("off")
        last_original, last_normalized = image, normalized
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "normalizacao_pixels.png", dpi=150)
    plt.close(fig)
    print("\\nValores originais: mínimo =", last_original.min(), "| máximo =", last_original.max())
    print("Valores normalizados: mínimo =", last_normalized.min(), "| máximo =", last_normalized.max())
    print("Gráficos de pré-processamento salvos em:", OUTPUT_DIR)
