from src.utils import load_pokemon_dataset
import numpy as np


def compute_single_linkage(cluster1, cluster2):
    dists = [np.linalg.norm(np.array(p1) - np.array(p2))
             for p1 in cluster1 for p2 in cluster2]
    return min(dists)


def compute_complete_linkage(cluster1, cluster2):
    dists = [np.linalg.norm(np.array(p1) - np.array(p2))
             for p1 in cluster1 for p2 in cluster2]
    return max(dists)


def compute_average_linkage(cluster1, cluster2):
    dists = [np.linalg.norm(np.array(p1) - np.array(p2))
             for p1 in cluster1 for p2 in cluster2]
    return sum(dists) / len(dists)


def compute_centroid_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)


def compute_ward_linkage(cluster1, cluster2):
    n1 = len(cluster1)
    n2 = len(cluster2)
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    centroid_dist = np.linalg.norm(centroid1 - centroid2)
    factor = np.sqrt((2 * n1 * n2) / (n1 + n2))
    return factor * centroid_dist


cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1, cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1, cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1, cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1, cluster2))
print("similaridade ligação de Ward: ", compute_ward_linkage(cluster1, cluster2))