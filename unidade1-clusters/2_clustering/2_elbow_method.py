import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.ma.core import append
#Import kmeans from scipy
from scipy.cluster.vq import kmeans
from src.utils import load_comic_con_dataset

comic_con = load_comic_con_dataset()

distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], i)
# append distortion on list distortions
    distortions.append(distortion)

# Create a DataFrame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()