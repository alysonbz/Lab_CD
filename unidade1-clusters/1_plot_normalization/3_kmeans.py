import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df

# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

df= loadpokemon_dataset_df()

# Compute cluster centers
centroids,data = kmeans(df,3)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
sns.scatterplot(x = 'x_cor', y = 'y_cor', hue='cluster labels', data=df)
plt.show()