import numpy as np
import random
from src.utils import load_pokemon_dataset

df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster, coords):
    # Escolhe pontos reais do dataset para iniciar os centroides,
    # garantindo que eles tenham o número correto de dimensões (N)
    coord_list = random.sample(coords, num_of_cluster)
    label_list = list(range(0, num_of_cluster))
    return coord_list, label_list


def create_points(df):
    # Seleciona apenas as colunas numéricas do DataFrame para criar os pontos em N dimensões
    # Se quiser usar colunas específicas, substitua por: df[['col1', 'col2', 'col3']].values.tolist()
    numeric_df = df.select_dtypes(include=[np.number])
    coords = numeric_df.values.tolist()
    return coords


def dist_euclidian(p1, p2):
    # Calcula a distância euclidiana para N dimensões usando NumPy
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p1 - p2)
    return dist


def kmeans(df, num_of_cluster):
    coords = create_points(df)
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster, coords)

    increase_cluster = True
    old_coord_label = []

    while increase_cluster == True:
        coord_label = []

        for coord in coords:
            coord_label.append(None)

            # Passo 1: Encontrar o centroide mais próximo (funciona para N dimensões)
            min_dist = float('inf')
            closest_label = None

            for i in range(len(centroids)):
                dist = dist_euclidian(coord, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    closest_label = centroids_labels[i]

            coord_label[-1] = closest_label

        # Passo 2: Condição de parada (Convergência)
        if coord_label == old_coord_label:
            increase_cluster = False
        else:
            old_coord_label = coord_label.copy()

            # Passo 3: Recalcular os centroides fazendo a média em cada uma das N dimensões
            new_centroids = []
            for label in centroids_labels:
                points_in_cluster = [coords[i] for i in range(len(coords)) if coord_label[i] == label]

                if len(points_in_cluster) > 0:
                    # O axis=0 faz com que o NumPy tire a média de cada coluna individualmente
                    mean_centroid = np.mean(points_in_cluster, axis=0).tolist()
                    new_centroids.append(mean_centroid)
                else:
                    # Se o cluster esvaziar, sorteia um ponto existente de N dimensões para reiniciar
                    new_centroids.append(random.choice(coords))

            centroids = new_centroids

    return coord_label


# Execução do K-Means em N dimensões
num_of_clusters = 3
labels_finais = kmeans(df, num_of_clusters)

# Adiciona os resultados ao DataFrame para conferência
df['cluster_n_dimension'] = labels_finais
print("Clusters atribuídos com sucesso para N dimensões!")
print(df.head())
