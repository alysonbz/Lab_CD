import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    
from src.utils import loadpokemon_dataset_df

# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

df= loadpokemon_dataset_df()

# Compute cluster centers
centroids,data = kmeans(df, 2)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
sns.scatterplot(x = df['x'], y = df['y'], hue=df['cluster_labels'], data=df)
plt.show()