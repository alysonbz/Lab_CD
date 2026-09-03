import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils import load_pokemon_dataset
import random
import matplotlib.pyplot as plt


df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = range(0,num_of_cluster)
    #preecher a lista com quatro coordenadas aleatótias.
    for i in range(num_of_cluster):
        coord_list.append([random.randint(min(df[0]), max(df[0])), random.randint(min(df[1]), max(df[1]))])
    return coord_list , label_list

def create_points(df):
    coords = [] # [ [x1,y1], [x2,y2] , [x3,y3]....]
    for i in range(len(df[0])):
        coords.append([df[0][i], df[1][i]])
    return coords

def dist_euclidian(p1,p2):
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i])**2
    dist = dist**0.5
    return dist

def kmeans(df,num_of_cluster):

    centroids ,  centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coords  = create_points(df)
    print(f"Centroids length: {len(centroids)}")
    print(f"Coords length: {len(coords)}")

    coord_label = []  #
    while increase_cluster == True:
        for coord in coords:
            coord_label.append(None)
            min_dist = float('inf')
            for i in range(len(centroids)):
                dist = dist_euclidian(coord, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    coord_label[-1] = centroids_labels[i]
        new_centroids = []
        for i in range(len(centroids)):
            cluster_points = [coords[j] for j in range(len(coords)) if coord_label[j] == centroids_labels[i]]
            if len(cluster_points) > 0:
                new_centroid = [sum(x)/len(x) for x in zip(*cluster_points)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])
        if new_centroids == centroids:
            increase_cluster = False
        else:
            centroids = new_centroids
            coord_label = []
    print(f"Coord label length: {len(coord_label)}")

    return  coord_label


num_of_clusters = 2

clusters = kmeans(df,num_of_clusters)

plt.scatter(df[0], df[1], c=clusters)
plt.title('K-means Clustering')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()