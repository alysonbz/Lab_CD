
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from .config import (
    CLEAN_DIR,
    OUTPUT_DIR,
    CLASSES,
    IMAGE_EXTENSIONS
)


def foreground_mask(imagem):
    """
    Gera uma máscara binária do objeto em primeiro plano.

    Parâmetros:
        imagem: imagem RGB em formato numpy.

    Retorno:
        Máscara binária uint8:
        255 = objeto
        0 = fundo
    """

    if imagem is None:
        raise ValueError("Imagem inválida para segmentação.")

    # Conversão para escala de cinza
    gray = cv2.cvtColor(
        imagem,
        cv2.COLOR_RGB2GRAY
    )

    # Suavização para reduzir ruídos
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Limiarização automática de Otsu
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Identificação da intensidade média das bordas
    bordas = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    intensidade_fundo = np.median(bordas)

    # Seleciona a polaridade mais adequada
    if intensidade_fundo > np.median(gray):
        mascara = cv2.bitwise_not(otsu)
    else:
        mascara = otsu

    # Operações morfológicas
    kernel = np.ones((5, 5), np.uint8)

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    # Identifica os contornos externos
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Mantém somente o maior componente
    if contornos:
        maior_contorno = max(
            contornos,
            key=cv2.contourArea
        )
        mascara_final = np.zeros_like(mascara)
        cv2.drawContours(
            mascara_final,
            [maior_contorno],
            -1,
            255,
            thickness=cv2.FILLED
        )
        return mascara_final

    # Caso não seja encontrado um objeto
    return np.ones_like(gray, dtype=np.uint8) * 255


def run():
    fig, axes = plt.subplots(
        len(CLASSES),
        3,
        figsize=(12, 20)
    )
    for row, class_name in enumerate(CLASSES):
        folder = CLEAN_DIR / class_name
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(
                f"Nenhuma imagem encontrada para {class_name}"
            )
        image = np.array(
            Image.open(files[0]).convert("RGB")
        )
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
        mascara = foreground_mask(image)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"{class_name} - Original"
        )
        axes[row, 1].imshow(
            gray,
            cmap="gray"
        )
        axes[row, 1].set_title(
            "Escala de cinza"
        )
        axes[row, 2].imshow(
            mascara,
            cmap="gray"
        )
        axes[row, 2].set_title(
            "Segmentação - Otsu"
        )
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "segmentacao_otsu.png",
        dpi=150
    )
    plt.close(fig)
    print(
        "Resultado da segmentação salvo em:",
        OUTPUT_DIR / "segmentacao_otsu.png"
    )


if __name__ == "__main__":
    run()
