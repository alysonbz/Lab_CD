import matplotlib.pyplot as plt

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils import load_comic_con_dataset

# Import the dendrogram function
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

df = load_comic_con_dataset()

distance_matrix = linkage(df, method='ward', metric='euclidean')

# Create a dendrogram
dn = dendrogram(distance_matrix)

# Display the dendrogram
plt.show()