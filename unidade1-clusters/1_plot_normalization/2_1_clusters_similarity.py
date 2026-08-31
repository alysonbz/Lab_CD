from src.utils import load_pokemon_dataset
import numpy as np


def compute_single_linkage(cluster1, cluster2):
    min_dist = float('inf')
    for p1 in cluster1:
        for p2 in cluster2:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def compute_complete_linkage(cluster1, cluster2):
    max_dist = 0
    for p1 in cluster1:
        for p2 in cluster2:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist > max_dist:
                max_dist = dist
    return max_dist


def compute_average_linkage(cluster1, cluster2):
    total_dist = 0
    count = 0
    for p1 in cluster1:
        for p2 in cluster2:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            total_dist += dist
            count += 1
    return total_dist / count


def compute_centroid_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)


def compute_ward_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    centroid_merged = np.mean(np.array(cluster1 + cluster2), axis=0)

    n1 = len(cluster1)
    n2 = len(cluster2)

    dist = (n1 * n2) / (n1 + n2) * np.linalg.norm(centroid1 - centroid2) ** 2
    return np.sqrt(dist)


cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]

print("similaridade ligação simples: ",
      compute_single_linkage(cluster1, cluster2))
print("similaridade ligação completa: ",
      compute_complete_linkage(cluster1, cluster2))
print("similaridade ligação média: ",
      compute_average_linkage(cluster1, cluster2))
print("similaridade pelo método do centroide: ",
      compute_centroid_linkage(cluster1, cluster2))
print("similaridade ligação simples: ",
      compute_ward_linkage(cluster1, cluster2))
