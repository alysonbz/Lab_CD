### Aula 01 ####

# Single Linkage (Ligação Simples) = A "menor distância" entre qualquer ponto do cluster 1 e qualquer ponto do cluster 2.

# Complete Linkage (Ligação Completa) = A "maior distância" entre qualquer ponto do cluster 1 e qualquer ponto do cluster 2.

# Avarage Linkage (Ligação Média): A média de todas as distâncias entre os pontos do cluster 1 e do cluster.

#### Aula 02 - Kmeans ###
# Testa a implementação com scipy.
from scipy.cluster.vq import kmeans, vq
from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd

import random
random.seed((1000, 2000))




# Necessidade de normalizar dados = variabilidade, grandezas diferentes, etc.
# Ex. Normalização logarítimica.

