from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from src.utils import load_fifa_dataset

fifa = load_fifa_dataset()

# Fit the data into a k-means algorithm (e.g., using 3 clusters)
cluster_centers, _ = kmeans(fifa[['sliding_tackle', 'aggression']], 3)

# Assign cluster labels using vector quantization (vq)
fifa['cluster_labels'], _ = vq(fifa[['sliding_tackle', 'aggression']], cluster_centers)

# Display cluster centers by calculating mean values per cluster
print(fifa[['sliding_tackle', 'aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='sliding_tackle', y='aggression', hue='cluster_labels', data=fifa)
plt.show()