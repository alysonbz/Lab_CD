import numpy as np

def euclidian_distance(point1, point2):
     return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def compute_single_linkage(cluster1,cluster2):
     min_distance = 0
     for point1 in cluster1:
          for point2 in cluster2:
               distance = euclidian_distance(point1,point2)
               if min_distance == 0 or distance < min_distance:
                    min_distance = distance
     return min_distance

def compute_complete_linkage(cluster1, cluster2):
     max_distance = 0
     for point1 in cluster1:
          for point2 in cluster2:
               distance = euclidian_distance(point1,point2)
               if max_distance == 0 or distance > max_distance:
                    max_distance = distance
     return max_distance

def compute_average_linkage(cluster1, cluster2):
     total_distance = 0
     count = 0
     for point1 in cluster1:
          for point2 in cluster2:
               total_distance += euclidian_distance(point1,point2)
               count += 1
     return total_distance / count if count > 0 else 0

def compute_centroid_linkage(cluster1,cluster2):
     centroid1 = np.mean(cluster1, axis=0)
     centroid2 = np.mean(cluster2, axis=0)
     distance = euclidian_distance(centroid1, centroid2)
     return distance

def compute_ward_linkage(cluster1,cluster2):
     centroid1 = np.mean(cluster1, axis=0)
     centroid2 = np.mean(cluster2, axis=0)
     distance = euclidian_distance(centroid1, centroid2)
     return distance


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))



