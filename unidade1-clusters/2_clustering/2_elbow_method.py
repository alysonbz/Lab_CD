import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Import kmeans from scipy
from scipy.cluster.vq import kmeans
from src.utils import load_comic_con_dataset

comic_con = load_comic_con_dataset()

distortions = []
num_clusters = range(1, 7)


for i in num_clusters:
    cluster_centers, distortion = kmeans(comic_con, i)
    distortions.append(distortion)


elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()