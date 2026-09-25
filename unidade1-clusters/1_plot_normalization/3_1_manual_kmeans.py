import random
import pandas as pd
from src.utils import load_pokemon_dataset

x, y = load_pokemon_dataset()
df = pd.DataFrame({'x': x, 'y': y})


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = list(range(0, num_of_cluster))
    for _ in range(num_of_cluster):
        cx = random.uniform(df['x'].min(), df['x'].max())
        cy = random.uniform(df['y'].min(), df['y'].max())
        coord_list.append([cx, cy])
    return coord_list, label_list


def create_points(df):
    coords = [[row['x'], row['y']] for _, row in df.iterrows()]
    return coords


def dist_euclidian(p1, p2):
    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return dist


def kmeans(df, num_of_cluster, max_iterations=100):
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coords = create_points(df)
    coord_label = [None] * len(coords)
    iterations = 0

    while increase_cluster and iterations < max_iterations:
        new_labels = []
        for coord in coords:
            distances = [dist_euclidian(coord, c) for c in centroids]