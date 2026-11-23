import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .config import CLEAN_DIR, OUTPUT_DIR, CLASSES, IMAGE_EXTENSIONS

def run():
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(12, 20))
    for row, class_name in enumerate(CLASSES):
        folder = CLEAN_DIR / class_name
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise RuntimeError(f"Nenhuma imagem encontrada para a classe {class_name}")
        image = np.array(Image.open(files[0]).convert("RGB"))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        axes[row, 0].imshow(image); axes[row, 0].set_title(f"{class_name} - Original")
        axes[row, 1].imshow(gray, cmap="gray"); axes[row, 1].set_title("Escala de cinza")
        axes[row, 2].imshow(mask, cmap="gray"); axes[row, 2].set_title("Segmentação - Otsu")
        for ax in axes[row]: ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "segmentacao_otsu.png", dpi=150)
    plt.close(fig)
    print("Resultado da segmentação salvo em:", OUTPUT_DIR / "segmentacao_otsu.png")
