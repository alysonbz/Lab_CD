import pandas as pd
import os


def carregar_dados():
    # Caminho ajustado conforme sua estrutura
    metadata_path = 'archive/Chest_xray_Corona_Metadata.csv'
    df = pd.read_csv(metadata_path)

    # Separação de Treino e Teste baseada na coluna Dataset_type
    treino_df = df[df['Dataset_type'] == 'TRAIN']
    teste_df = df[df['Dataset_type'] == 'TEST']

    # X = Nomes das imagens, y = Labels
    X_train = treino_df['X_ray_image_name'].values
    y_train = treino_df['Label'].values

    X_test = teste_df['X_ray_image_name'].values
    y_test = teste_df['Label'].values

    print(f"{len(X_train)} treino, {len(X_test)} teste.")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    carregar_dados()