
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import (
    local_binary_pattern,
    hog,
    graycomatrix,
    graycoprops
)

from .config import OUTPUT_DIR
from .preprocessamento import preprocess_image
from .segmentacao import foreground_mask


def extrair_cor(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)

    caracteristicas = {}

    canais = [
        (0, "h", 18, [0, 180]),
        (1, "s", 16, [0, 256]),
        (2, "v", 16, [0, 256])
    ]

    for canal, nome, bins, intervalo in canais:
        hist = cv2.calcHist(
            [hsv], [canal], None, [bins], intervalo
        )
        hist = cv2.normalize(hist, hist).flatten()

        for i, valor in enumerate(hist):
            caracteristicas[f"cor_{nome}_{i}"] = float(valor)

    for i, nome in enumerate(["r", "g", "b"]):
        caracteristicas[f"rgb_{nome}_media"] = float(
            imagem[:, :, i].mean() / 255
        )
        caracteristicas[f"rgb_{nome}_desvio"] = float(
            imagem[:, :, i].std() / 255
        )

    return caracteristicas


def extrair_textura(imagem):
    cinza = cv2.cvtColor(
        imagem, cv2.COLOR_RGB2GRAY
    )

    # Local Binary Patterns
    lbp = local_binary_pattern(
        cinza, P=8, R=1, method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 12),
        density=True
    )

    caracteristicas = {
        f"lbp_{i}": float(valor)
        for i, valor in enumerate(hist)
    }

    # Gray Level Co-occurrence Matrix
    quantizada = (cinza // 16).astype(np.uint8)

    glcm = graycomatrix(
        quantizada,
        distances=[1, 2],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )

    propriedades = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for propriedade in propriedades:
        valores = graycoprops(glcm, propriedade).flatten()

        for i, valor in enumerate(valores):
            caracteristicas[
                f"glcm_{propriedade}_{i}"
            ] = float(valor)

    return caracteristicas


def extrair_forma(imagem):
    mascara = foreground_mask(imagem)

    altura, largura = mascara.shape

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    caracteristicas = {}

    if contornos:
        contorno = max(
            contornos,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        x, y, w, h = cv2.boundingRect(contorno)

        circularidade = (
            4 * np.pi * area / (perimetro ** 2 + 1e-9)
        )

        momentos = cv2.moments(contorno)

        hu = cv2.HuMoments(momentos).flatten()

        caracteristicas.update({
            "forma_area": float(area / (altura * largura)),
            "forma_perimetro": float(
                perimetro / (2 * (altura + largura))
            ),
            "forma_circularidade": float(circularidade),
            "forma_proporcao": float(w / (h + 1e-9))
        })

        for i, valor in enumerate(hu):
            caracteristicas[f"hu_{i}"] = float(
                -np.sign(valor) * np.log10(abs(valor) + 1e-30)
            )

    else:
        caracteristicas.update({
            "forma_area": 0,
            "forma_perimetro": 0,
            "forma_circularidade": 0,
            "forma_proporcao": 0
        })

        for i in range(7):
            caracteristicas[f"hu_{i}"] = 0

    return caracteristicas


def extrair_hog(imagem):
    cinza = cv2.cvtColor(
        imagem, cv2.COLOR_RGB2GRAY
    )

    vetor = hog(
        cinza,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return {
        f"hog_{i}": float(valor)
        for i, valor in enumerate(vetor)
    }


def extrair_caracteristicas(caminho, configuracao):
    imagem = preprocess_image(caminho)

    resultado = extrair_cor(imagem)

    if configuracao in ["cor_textura", "completa"]:
        resultado.update(extrair_textura(imagem))

    if configuracao == "completa":
        resultado.update(extrair_forma(imagem))
        resultado.update(extrair_hog(imagem))

    return resultado


def construir_dataframe(registros, configuracao):
    linhas = []

    for registro in registros:
        caminho = registro["path"]

        caracteristicas = extrair_caracteristicas(
            caminho,
            configuracao
        )

        caracteristicas["imagem_id"] = Path(
            caminho
        ).name

        caracteristicas["caminho"] = caminho
        caracteristicas["classe"] = registro["label"]

        linhas.append(caracteristicas)

    return pd.DataFrame(linhas)


def run():
    split_path = OUTPUT_DIR / "split.json"

    with open(split_path, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    registros = dados["train"] + dados["test"]

    configuracoes = {
        "cor": "dataframe_cor.csv",
        "cor_textura": "dataframe_cor_textura.csv",
        "completa": "dataframe_completo.csv"
    }

    for configuracao, nome_arquivo in configuracoes.items():

        print(f"\nExtraindo características: {configuracao}")

        df = construir_dataframe(
            registros,
            configuracao
        )

        df.to_csv(
            OUTPUT_DIR / nome_arquivo,
            index=False
        )

        print("Dimensões:", df.shape)
        print("Arquivo salvo:", nome_arquivo)

    print("\nExtração concluída!")


if __name__ == "__main__":
    run()
