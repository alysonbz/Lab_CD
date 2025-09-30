from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd

x_coordinates = [80.1, 93.1, 86.6, 99.5, 86.4, 9.5, 15.2, 3.4,
                 10.4, 23.4, 64.2, 50.8, 09.2, 62.5, 64.0]
y_coordinates = [87.2, 90.1, 95.6, 92.6, 92.4, 57.7, 47.0,
                 47.3, 59.1, 55.5, 26.2, 21.0, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate': x_coordinates,
                   'y_coordinate': y_coordinates})

Z = linkage(df, 'ward')
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')

sns.scatterplot(x='x_coordinate', y='y_coordinate',
                hue='cluster_labels', data=df)

plt.show()
