import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops


def extrair_atributos(caminho_img, label):
    # Verifica se o arquivo físico existe
    if not os.path.exists(caminho_img):
        return None

    # Tenta ler a imagem
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Redimensionamento para padronizar o cálculo
    img = cv2.resize(img, (128, 128))

    # Extração GLCM (Mineração de Atributos - Tópico 3)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    features = {
        'contraste': graycoprops(glcm, 'contrast')[0, 0],
        'homogeneidade': graycoprops(glcm, 'homogeneity')[0, 0],
        'energia': graycoprops(glcm, 'energy')[0, 0],
        'correlacao': graycoprops(glcm, 'correlation')[0, 0],  # Adicionei mais um atributo
        'media_pixels': np.mean(img),
        'desvio_padrao': np.std(img),
        'label': 1 if label == 'Pnemonia' else 0
    }
    return features


def gerar_csvs():
    # Caminho do metadados (ajuste se necessário)
    df_metadata = pd.read_csv('Chest_xray_Corona_Metadata.csv')

    # Montando o caminho base conforme sua descrição (pasta repetida)
    # O ".." volta um nível para sair da pasta AV2
    base_path = os.path.join('Coronahack-Chest-XRay-Dataset')

    for tipo in ['TRAIN', 'TEST']:
        print(f"\n--- Iniciando {tipo} ---")
        subset = df_metadata[df_metadata['Dataset_type'] == tipo]
        folder_name = tipo.lower()  # 'train' ou 'test'

        lista_final = []
        for _, row in subset.iterrows():
            # Construindo o caminho completo da imagem
            caminho_completo = os.path.join(base_path, folder_name, row['X_ray_image_name'])

            f = extrair_atributos(caminho_completo, row['Label'])

            if f is not None:
                lista_final.append(f)
            # Opcional: print para debug se quiser ver o progresso
            # print(f"Processado: {row['X_ray_image_name']}")

        if lista_final:
            output_file = f'atributos_{tipo.lower()}.csv'
            pd.DataFrame(lista_final).to_csv(output_file, index=False)
            print(f"Sucesso! {output_file} gerado com {len(lista_final)} linhas.")
        else:
            print(f"ERRO: Nenhuma imagem encontrada em: {os.path.join(base_path, folder_name)}")


if __name__ == "__main__":
    gerar_csvs()