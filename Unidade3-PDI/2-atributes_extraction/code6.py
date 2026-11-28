from collections import deque
import cv2
import numpy as np
from src.pdi_utils import load_soaps_image


def crescimento_regioes(imagem, semente, limiar=30):
    """Realiza o crescimento de regiões a partir de uma semente (x, y)

    utilizando a distância euclidiana de cores (BGR) como critério.
    """
    altura, largura, _ = imagem.shape

    # Máscara binária para marcar os pixels que pertencem ao objeto
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    visitados = np.zeros((altura, largura), dtype=bool)

    # Coordenada inicial (OpenCV usa [linha, coluna], ou seja, [y, x])
    x_inicial, y_inicial = semente
    cor_semente = imagem[y_inicial, x_inicial].astype(np.float32)

    # Fila para processamento iterativo (BFS)
    fila = deque([(x_inicial, y_inicial)])
    visitados[y_inicial, x_inicial] = True
    mascara[y_inicial, x_inicial] = 255

    # Vizinhança-8 (considera diagonais para evitar falhas nas bordas)
    vizinhos = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    while fila:
        x, y = fila.popleft()

        for dx, dy in vizinhos:
            nx, ny = x + dx, y + dy

            # Verifica se está dentro dos limites da imagem e ainda não foi visitado
            if 0 <= nx < largura and 0 <= ny < altura and not visitados[ny, nx]:
                visitados[ny, nx] = True

                cor_atual = imagem[ny, nx].astype(np.float32)
                # Calcula a diferença de cor em relação à semente
                diferenca = np.linalg.norm(cor_atual - cor_semente)

                if diferenca <= limiar:
                    mascara[ny, nx] = 255
                    fila.append((nx, ny))

    # Aplica a máscara na imagem original: mantém a cor onde máscara == 255 e preto no resto
    resultado = cv2.bitwise_and(imagem, imagem, mask=mascara)
    return resultado


def evento_clique(event, x, y, flags, param):
    """Callback executado ao clicar com o mouse na janela da imagem."""
    if event == cv2.EVENT_LBUTTONDOWN:
        imagem_original = param["imagem"]
        limiar = param["limiar"]

        print(f"Semente capturada nas coordenadas (x={x}, y={y}). Processando...")

        # Executa a segmentação
        imagem_segmentada = crescimento_regioes(
            imagem_original, (x, y), limiar=limiar
        )

        # Exibe o resultado final
        cv2.imshow("Objeto Segmentado", imagem_segmentada)


def main():
    # Carrega a imagem usando a função do seu projeto
    img = load_soaps_image()

    # Caso a função retorne em RGB (padrão matplotlib/skimage), converte para BGR (padrão OpenCV)
    # Se as cores ficarem invertidas, comente a linha abaixo:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Pré-processamento opcional: filtro Gaussiano leve para suavizar ruídos/texturas
    img_suavizada = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    nome_janela = "Clique no objeto para segmentar"
    cv2.namedWindow(nome_janela)

    # Ajuste o 'limiar' (ex: entre 20 e 60) dependendo da variação de cor/sombra do objeto
    parametros = {"imagem": img_suavizada, "limiar": 35}
    cv2.setMouseCallback(nome_janela, evento_clique, parametros)

    cv2.imshow(nome_janela, img_bgr)
    print(
        "Clique com o botão esquerdo sobre o objeto desejado. Pressione qualquer tecla para sair."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
