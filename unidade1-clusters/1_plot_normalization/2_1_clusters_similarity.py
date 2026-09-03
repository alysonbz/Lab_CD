from src.utils import load_pokemon_dataset
import numpy as np



# Distância Euclidiana
def euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_single_linkage(cluster1,cluster2):
    min_dist = float('inf')

    for p1 in cluster1:
        for p2 in cluster2:
            dist = euclidiana(p1, p2)
            if dist < min_dist:
                min_dist = dist

    return min_dist

def compute_complete_linkage(cluster1, cluster2):
    max_dist = 0

    for p1 in cluster1:
        for p2 in cluster2:
            dist = euclidiana(p1, p2)
            if dist > max_dist:
                max_dist = dist

    return max_dist

def compute_average_linkage(cluster1, cluster2):
    distances = []

    for p1 in cluster1:
        for p2 in cluster2:
            distances.append(euclidiana(p1, p2))

    return np.mean(distances)

def compute_centroid_linkage(cluster1,cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

    return euclidiana(centroid1, centroid2)

def compute_ward_linkage(cluster1,cluster2):
    # Centroides dos dois clusters
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

    # Tamanho dos clusters
    n1 = len(cluster1)
    n2 = len(cluster2)

    # Distância euclidiana entre os centroides
    dist_centroides = euclidiana(centroid1, centroid2)

    # Critério de Ward
    ward_distance = (n1 * n2) / (n1 + n2) * (dist_centroides ** 2)

    return ward_distance


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))



