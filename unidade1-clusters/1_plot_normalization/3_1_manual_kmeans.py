import numpy as np
import random
from src.utils import loadpokemon_dataset_df


df = loadpokemon_dataset_df()


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = list(range(0, num_of_cluster))
    for i in range(num_of_cluster):
        coord_list.append([random.uniform(0, 100), random.uniform(0, 100)])
    return coord_list, label_list


def create_points(df):
    coords = []
    for index, row in df.iterrows():
        coords.append([row['x'], row['y']])
    return coords


def dist_euclidian(p1, p2):
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def kmeans(df, num_of_cluster):
    centroids,  centroids_labels = set_random_cluster_coordinate(
        num_of_cluster)
    increase_cluster = True
    coords = create_points(df)
    coord_label = []
    iterations = 0
    max_iterations = 100

    while increase_cluster == True and iterations < max_iterations:
        iterations += 1
        old_centroids = [c[:] for c in centroids]
        coord_label = []

        # Atribuir cada ponto ao centroide mais proximo
        for coord in coords:
            min_dist = float('inf')
            closest_centroid = 0
            for i, centroid in enumerate(centroids):
                d = dist_euclidian(coord, centroid)
                if d < min_dist:
                    min_dist = d
                    closest_centroid = i
            coord_label.append(closest_centroid)

        # Recalcular centroides
        for i in range(num_of_cluster):
            cluster_points = [coords[j]
                              for j in range(len(coords)) if coord_label[j] == i]
            if cluster_points:
                centroids[i] = np.mean(cluster_points, axis=0).tolist()

        # Verificar convergencia
        if np.allclose(old_centroids, centroids):
            increase_cluster = False

    return coord_label


num_of_clusters = 2
