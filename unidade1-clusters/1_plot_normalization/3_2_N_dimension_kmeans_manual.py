from src.utils import load_pokemon_dataset
import numpy as np
import random


df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    for i in range(num_of_cluster):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        coord_list.append([x, y])
    label_list = range(0,num_of_cluster)
    #preecher a lista com quatro coordenadas aleatótias.
    return [[0,0],[30,30]] , label_list

def create_points(df):
    coords = [] # [ [x1,y1], [x2,y2] , [x3,y3]....]
    for i in range(len(df[0])):
        coords.append([df[0][i], df[1][i]])
    return coords

def dist_euclidian(p1,p2):
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

def kmeans(df,num_of_cluster):

    centroids ,  centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coords  = create_points(df)
    coord_label = []  #
    while increase_cluster == True:
        for coord in coords:
            dist0 = dist_euclidian(coord,centroids[0])
            dist1 = dist_euclidian(coord,centroids[1])
            if dist0 < dist1:
                coord_label.append(centroids_labels[0])
            else:
                coord_label.append(centroids_labels[1])
            ##algoritmo
        increase_cluster = False


    return  coord_label


num_of_clusters = 2
print(set_random_cluster_coordinate(num_of_clusters))
print(create_points(df))
print(kmeans(df,num_of_clusters))