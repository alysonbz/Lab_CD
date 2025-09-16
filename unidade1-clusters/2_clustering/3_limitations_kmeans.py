from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from src.utils import load_fifa_dataset


fifa = load_fifa_dataset()
# Fit the data into a k-means algorithm
cluster_centers,_ =___

# Assign cluster labels
fifa['cluster_labels'], _ =___

# Display cluster centers
print(fifa[['sliding_tackle','aggression', 'cluster_labels']].groupby(__).__())

# Create a scatter plot through seaborn
___
plt.show()