import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df
df= loadpokemon_dataset_df()


# Import linkage and fcluster functions
from scipy.cluster.hierarchy import ____, ____

# Use the linkage() function to compute distance
Z = ____(____, 'ward')

# Generate cluster labels
df['cluster_labels'] = ____(____, ____, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x=____, y=____, hue=____, data=df)
plt.show()