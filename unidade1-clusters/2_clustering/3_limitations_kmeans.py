from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    
from src.utils import load_fifa_dataset


fifa = load_fifa_dataset()
# Fit the data into a k-means algorithm
cluster_centers,_ = kmeans(fifa[['sliding_tackle','aggression']], 3)

# Assign cluster labels
fifa['cluster_labels'], _ = vq(fifa[['sliding_tackle','aggression']], cluster_centers)

# Display cluster centers
print(fifa[['sliding_tackle','aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='sliding_tackle', y='aggression', hue='cluster_labels', data=fifa)
plt.show()