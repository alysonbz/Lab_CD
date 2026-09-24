from src.utils import load_pokemon_dataset
import random
import math


df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = range(0, num_of_cluster)

    # preencher a lista com coordenadas aleatórias
    for _ in range(num_of_cluster):
        coord_list.append([random.uniform(0, 10), random.uniform(0, 10)])

    return coord_list, label_list


def create_points(df):
    coords = []  # [ [x1,y1], [x2,y2] , [x3,y3]....]

    for _, row in df.iterrows():
        coords.append([row['x'], row['y']])

    return coords


def dist_euclidian(p1, p2):
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def kmeans(df, num_of_cluster):

    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coords = create_points(df)
    coord_label = []

    while increase_cluster == True:
        coord_label = []

        for coord in coords:
            min_dist = float('inf')
            best_label = None

            for i, centroid in enumerate(centroids):
                dist = dist_euclidian(coord, centroid)

                if dist < min_dist:
                    min_dist = dist
                    best_label = centroids_labels[i]

            coord_label.append(best_label)

        increase_cluster = False

    return coord_label


num_of_clusters = 2