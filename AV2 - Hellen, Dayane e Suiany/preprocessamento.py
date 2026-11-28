"""
Funções das Etapas 2 (pré-processamento) e 3 (segmentação).

Uso na Etapa 4:
    from preprocessamento import carregar_rgb, preprocessar, normalizar, segmentar, recortar_fruta

    img = carregar_rgb(caminho)                    # RGB, uint8, 100x100x3
    img = preprocessar(img, usar_clahe=False)      # CLAHE é opcional
    mascara = segmentar(img, "uniao")              # 0 = fundo, 255 = fruta
    fruta = recortar_fruta(img, mascara)           # fruta sobre fundo branco
    img_norm = normalizar(img)                     # float32 em [0, 1]

Decisões (detalhes nos notebooks 02 e 03):
- Sem redimensionamento (imagens já são 100x100).
- Sem filtro de ruído (destrói a textura da casca).
- CLAHE só no canal L do Lab, e opcional (validar na Etapa 5).
- Segmentação: união (OR) de Otsu no canal S do HSV com Otsu em tons de cinza,
  seguida de fechamento morfológico 5x5 e preenchimento do maior contorno.
- Cuidado: a máscara pode falhar em regiões muito claras (reflexos, cascas claras).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------
# Etapa 2 - Pré-processamento
# ---------------------------------------------------------------

def carregar_rgb(caminho):
    """Lê a imagem com OpenCV e converte de BGR para RGB."""
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def clahe_lab(img_rgb, clip=2.0, grade=(4, 4)):
    """CLAHE aplicado apenas no canal L do Lab (preserva a cor)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grade)
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2RGB)


def preprocessar(img_rgb, usar_clahe=False):
    """Pré-processamento da Etapa 2.
    Entrada: imagem RGB uint8. Saída: imagem RGB uint8 (CLAHE opcional)."""
    if usar_clahe:
        return clahe_lab(img_rgb)
    return img_rgb


def normalizar(img_uint8):
    """Converte uint8 (0-255) para float32 (0-1).
    Faça as conversões de espaço de cor (HSV, Lab) ANTES de normalizar."""
    return img_uint8.astype(np.float32) / 255.0


def desnormalizar(img_float):
    """Operação inversa de normalizar()."""
    return np.round(img_float * 255.0).astype(np.uint8)


# ---------------------------------------------------------------
# Etapa 3 - Segmentação
# ---------------------------------------------------------------

def mascara_otsu_cinza(img_rgb):
    """Máscara (255 = fruta) por Otsu em tons de cinza.
    Retorna (limiar, máscara)."""
    cinza = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


def mascara_otsu_saturacao(img_rgb):
    """Máscara (255 = fruta) por Otsu no canal S do HSV.
    Retorna (limiar, máscara)."""
    s = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[..., 1]
    return cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def limpar_mascara(mascara, tam_kernel=5):
    """Fechamento morfológico + preenchimento do maior contorno externo."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_kernel, tam_kernel))
    fechada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    limpa = np.zeros_like(mascara)
    if len(contornos) == 0:
        return limpa
    maior = max(contornos, key=cv2.contourArea)
    cv2.drawContours(limpa, [maior], -1, 255, thickness=cv2.FILLED)
    return limpa


def segmentar(img_rgb, metodo="uniao"):
    """Segmentação completa. metodo: 'S' (só canal S) ou 'uniao' (S OR cinza).
    O método final adotado na Etapa 3 é 'uniao'."""
    _, masc = mascara_otsu_saturacao(img_rgb)
    if metodo == "uniao":
        _, masc_c = mascara_otsu_cinza(img_rgb)
        masc = cv2.bitwise_or(masc, masc_c)
    return limpar_mascara(masc)


def recortar_fruta(img_rgb, mascara):
    """Mantém só os pixels da fruta e pinta o resto de branco."""
    saida = np.full_like(img_rgb, 255)
    saida[mascara > 0] = img_rgb[mascara > 0]
    return saida