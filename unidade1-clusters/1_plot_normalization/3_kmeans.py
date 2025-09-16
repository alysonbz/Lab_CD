import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df

# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

df= loadpokemon_dataset_df()

# Compute cluster centers
centroids,data = __(__,__)

# Assign cluster labels
df['cluster_labels'], _ = __(__,__)

# Plot the points with seaborn
sns.__(x = __, y = __, hue=__, data=__)
plt.show()