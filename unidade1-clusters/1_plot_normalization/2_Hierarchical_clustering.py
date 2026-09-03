import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils import loadpokemon_dataset_df
df= loadpokemon_dataset_df()


# Import linkage and fcluster functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function to compute distance
Z = linkage(df[['x', 'y']], 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()