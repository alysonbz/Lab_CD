import matplotlib.pyplot as plt
import seaborn as sns
#Import kmeans from scipy
---

from src.utils import load_comic_con_dataset

comic_con = load_comic_con_dataset()

distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in ____:
    cluster_centers, distortion = ____
# append distortion on list distortions
    ___

# Create a DataFrame with two lists - num_clusters, distortions
elbow_plot = ___({'num_clusters': ____, 'distortions': ____})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x=____, y=____, data = ____)
plt.xticks(num_clusters)
plt.show()