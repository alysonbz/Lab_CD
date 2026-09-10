import numpy as np
import random
from src.utils import load_pokemon_dataset

df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = range(0, num_of_cluster)
    # preecher a lista com quatro coordenadas aleatótias.
    for _ in label_list:
        coord_list.append([random.uniform(0, 100), random.uniform(0, 100)])
    return coord_list, label_list


def create_points(df):
    coords = []  # [ [x1,y1], [x2,y2] , [x3,y3]....]
    # Extrai os valores das colunas de características do Pokémon Dataset
    coords = df[['scaled_x', 'scaled_y']].values.tolist()
    return coords


def dist_euclidian(p1, p2):
    dist = 0
    # Calcula a distância euclidiana manual entre dois pontos [x, y]
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def kmeans(df, num_of_cluster):
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coords = create_points(df)
    coord_label = []  #

    # Variável auxiliar para rastrear a convergência (mudança de estado)
    old_coord_label = []

    while increase_cluster == True:
        coord_label = []  # Reinicia a lista a cada iteração para o novo mapeamento
        for coord in coords:
            coord_label.append(None)

            # Passo 1: Encontrar o centroide mais próximo para o ponto atual
            min_dist = float('inf')
            closest_label = None

            for i in range(len(centroids)):
                dist = dist_euclidian(coord, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    closest_label = centroids_labels[i]

            # Substitui o None adicionado pelo rótulo correto
            coord_label[-1] = closest_label

        # Passo 2: Condição de parada (Se as atribuições não mudarem, o algoritmo converge)
        if coord_label == old_coord_label:
            increase_cluster = False
        else:
            old_coord_label = coord_label.copy()

            # Passo 3: Recalcular a posição dos centroides tirando a média geométrica dos pontos
            new_centroids = []
            for label in centroids_labels:
                # Filtra todos os pontos atribuídos a este cluster específico
                points_in_cluster = [coords[i] for i in range(len(coords)) if coord_label[i] == label]

                if len(points_in_cluster) > 0:
                    # Calcula a média de X e Y
                    mean_x = np.mean([p[0] for p in points_in_cluster])
                    mean_y = np.mean([p[1] for p in points_in_cluster])
                    new_centroids.append([mean_x, mean_y])
                else:
                    # Se um cluster ficar vazio, gera um novo ponto aleatório para ele não sumir
                    new_centroids.append([random.uniform(0, 100), random.uniform(0, 100)])

            centroids = new_centroids

    return coord_label


num_of_clusters = 2
