import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .config import CLEAN_DIR, OUTPUT_DIR, CLASSES, IMAGE_EXTENSIONS

def preprocess_image(caminho, tamanho=(128, 128)):
    """
    Pré-processa uma imagem para extração de características.

    Etapas:
    1. Leitura da imagem.
    2. Conversão BGR para RGB.
    3. Redimensionamento preservando a proporção.
    4. Preenchimento com margens.
    5. Redução de ruído com filtro bilateral.
    6. Ajuste de contraste utilizando CLAHE no canal L do LAB.

    Retorna:
        Imagem RGB no formato uint8.
    """

    # 1. Leitura
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {caminho}")

    # 2. Conversão para RGB
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    altura, largura = imagem.shape[:2]
    largura_alvo, altura_alvo = tamanho

    # 3. Redimensionamento preservando proporção
    escala = min(
        largura_alvo / largura,
        altura_alvo / altura
    )

    nova_largura = max(1, int(largura * escala))
    nova_altura = max(1, int(altura * escala))

    imagem = cv2.resize(
        imagem,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA
    )

    # 4. Preenchimento com margens
    canvas = np.full(
        (altura_alvo, largura_alvo, 3),
        255,
        dtype=np.uint8
    )

    x = (largura_alvo - nova_largura) // 2
    y = (altura_alvo - nova_altura) // 2

    canvas[
        y:y + nova_altura,
        x:x + nova_largura
    ] = imagem

    imagem = canvas

    # 5. Redução moderada de ruído
    imagem = cv2.bilateralFilter(
        imagem,
        d=5,
        sigmaColor=35,
        sigmaSpace=35
    )

    # 6. CLAHE no canal de luminosidade
    lab = cv2.cvtColor(
        imagem,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    imagem = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return imagem

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
