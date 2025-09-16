# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 1

### Questão 1

[1_cluster_plot.py](1_cluster_plot.py)

#### Plotando conjunto de dados

Nesta questão você visualizar o conjunto de dados.

#### Instruções:

1)  Importe `` pyplot `` da matplotlib .
   
2)  Chame o método ``scatter``  através do objeto plt . 

3)  Mostre a distribuição dos pontos com o método `` show`` .

### Questão 2

[2_Hierarchical_clustering.py](2_Hierarchical_clustering.py)

#### Clusterização Hierárquica

Nesta questão você vai aplicar uma clusterização hierárquica.

#### Instruções 

1) Importe o módulo ``linkage `` e ``fcluster``.
2) utilize a função linkage com o método ward.
3) Crie uma nova coluna no dataframe df chamada ``cluster_labels``. Adicione nessa coluna 
as labels dos clusters que serão calculados com a função ``fcluster`` passe para função os argumentos Z, que representa a matriz de distâncias, 
a quantadidade de cluster defina igual 2.
4) Utilizando o módulo ``seaborn`` realize um ``scatterplot``. Defina os arguentos ``x`` = x, ``y``=y e ``hue`` = cluster_labels e ``data`` = df.


### Questão 2.1

[2_1_clusters_similarity.py](2_1_clusters_similarity.py)

#### Cálculo da similaridade de clusters

Nesta questão você vai realizar um cálculo para determinar a similaridade entre clusters.

#### Instruções

1) Com a função ``compute_single_linkage`` calcule a ligação simples.
2) Com a função ``compute_complete_linkage`` calcule a ligação completa
3) Com a função ``compute_average_linkage`` calcule a ligação média.
4) Com a função  ``compute_centroid_linkage`` calcule o método do centroide.
5) Com função ``compute_ward_linkage`` calcule o método de ward. 

dica: você pode implementar funções auxiliares além das descritas.


### Questão 3

[3_kmeans.py](3_kmeans.py)

#### Custerização com Kmeans

Nesta questão você vai realizar uma clusterização com o kmeans

#### Instruções

1) Instancie o kmeans com 2 clusters para o dataframe df
2) Em df['cluster_labels'] adicione as labels da clusterização usando a função  ``vq`` que recebe os argumentos df e centroids
3) Plote os dados com seaborn.

### Questão 3.1

[3_1_manual_kmeans.py](3_1_manual_kmeans.py)

#### Custerização com Kmeans manual

Nesta questão você vai realizar uma clusterização com o kmeans

#### Instruções

1) observando as funções, implemente manualmente a clusterização via k_means

### Questão 4

[4_scaled_data.py](4_scaled_data.py)

#### Normalização whiten

Nesta questão você deve observar os efeitos da normalização pelo divisão pelo desvio padrão

#### Instruções 

1) Utilizando a função ``whiten()`` aplique a normalização no dataset.
2) plote em linha os dados originais sem a normalização.
3) plote em linha os dados normalizados.
4) Exiba as diferenças.


### Questão 5

[5_small_number_norm.py](5_small_number_norm.py)

#### Normalização whiten com numeros proximos de zero

Nesta questão você deve observar os efeitos da normalização pelo divisão pelo desvio padrão em numeros proximos de zero

#### Instruções 

1) Utilizando a função ``whiten()`` aplique a normalização no dataset.
2) plote em linha os dados originais sem a normalização.
3) plote em linha os dados normalizados.
4) Exiba as diferenças.


### Questão 6

[6_log_normalization.py](6_log_normalization.py)

#### Normalização Logarítimica

Nesta questão você deve observar os efeitos da normalização logarítimica

#### Instruções 

1) Utilizando a função ``describe()`` verifique as características estatística do dataset ``wine``.
2) Na coluna``Proline``, aplique a normalizaçao logarítmica e atribua o resultado para uma nova coluna, ``Proline_log``.
3) Exiba a variância da coluna ``Proline``.
4) Print a variância da coluna ``Proline_log``.


### Questão 7

[7_scaling_data.py](7_scaling_data.py)

#### Normalização StandardScaler

Nesta questão você deve observar os efeitos da normalização por media zer e variância unitária

#### Instruções 

1) Importe o módulo ``StandardScaler``.
3) Instancie StandardScaler .
4) Em X_norm inclua os dados normalizados.
5) Print a variância de X
6) Print a variância de X_Norm

