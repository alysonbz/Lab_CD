


import matplotlib.pyplot as plt
from src.utils import load_comic_con_dataset
from scipy.cluster.hierarchy import dendrogram, linkage

df = load_comic_con_dataset()

# Cria a matriz de distâncias usando as colunas 'x_scaled' e 'y_scaled'
data = df[['x_scaled', 'y_scaled']]

# Cria a matriz de linkage com método 'ward' e distância euclidiana
distance_matrix = linkage(data, method='ward', metric='euclidean')

# Plota o dendrograma
dn = dendrogram(distance_matrix)

# Exibe o dendrograma
plt.show()
