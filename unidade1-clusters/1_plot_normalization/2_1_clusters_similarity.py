from src.utils import load_pokemon_dataset
import numpy as np


def compute_single_linkage(cluster1,cluster2):
    links = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            links.append(np.sqrt((cluster1[i][0] - cluster2[j][0]) ** 2 + (cluster1[i][1] - cluster2[j][1]) ** 2))
    return min(links)

def compute_complete_linkage(cluster1, cluster2):
    links = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            links.append(np.sqrt((cluster1[i][0] - cluster2[j][0]) ** 2 + (cluster1[i][1] - cluster2[j][1]) ** 2))
    return max(links)

def compute_average_linkage(cluster1, cluster2):
    links = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            links.append(np.sqrt((cluster1[i][0] - cluster2[j][0]) ** 2 + (cluster1[i][1] - cluster2[j][1]) ** 2))
    return np.mean(links)

def compute_centroid_linkage(cluster1,cluster2):
    centroide1 = np.mean(cluster1, axis=0)
    centroide2 = np.mean(cluster2, axis=0)
    return np.sqrt((centroide1[0]-centroide2[0])**2 + (centroide1[1]-centroide2[1])**2)
def compute_ward_linkage(cluster1,cluster2):
    centroide1 = np.mean(cluster1, axis=0)
    centroide2 = np.mean(cluster2, axis=0)
    d = ((len(cluster1)*len(cluster2))/(len(cluster1)+len(cluster2)))*(sum((centroide1-centroide2)**2))
    return d


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação ward: ", compute_ward_linkage(cluster1,cluster2))