from src.utils import load_pokemon_dataset
import numpy as np


def compute_single_linkage(cluster1,cluster2):
    distancias = []

    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(np.array(ponto1) - np.array(ponto2))
            distancias.append(distancia)

    return min(distancias)

def compute_complete_linkage(cluster1, cluster2):
    distancias = []

    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(np.array(ponto1) - np.array(ponto2))
            distancias.append(distancia)

    return max(distancias)

def compute_average_linkage(cluster1, cluster2):
    distancias = []

    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(np.array(ponto1) - np.array(ponto2))
            distancias.append(distancia)

    return np.mean(distancias)

def compute_centroid_linkage(cluster1, cluster2):
    centroide1 = np.mean(cluster1, axis=0)
    centroide2 = np.mean(cluster2, axis=0)

    return np.linalg.norm(centroide1 - centroide2)

def compute_ward_linkage(cluster1, cluster2):
    quantidade1 = len(cluster1)
    quantidade2 = len(cluster2)

    centroide1 = np.mean(cluster1, axis=0)
    centroide2 = np.mean(cluster2, axis=0)

    distancia = np.linalg.norm(centroide1 - centroide2)

    return np.sqrt(2 * quantidade1 * quantidade2 / (quantidade1 + quantidade2)) * distancia


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))