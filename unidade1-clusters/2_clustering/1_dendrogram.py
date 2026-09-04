import matplotlib.pyplot as plt
from src.utils import load_comic_con_dataset

# Import the dendrogram function
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

df = load_comic_con_dataset()

distance_matrix = linkage(df, method='ward')

# Create a dendrogram
dn = dendrogram(distance_matrix, orientation='right')

# Display the dendogram
plt.show()