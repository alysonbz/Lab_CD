# AV2 — Classificação de imagens com extração de características (PlantVillage)

**Disciplina:** Ciência de Dados  |  **Equipe:** Kauê Cruz e Abimael

**Dataset:** PlantVillage (repositório oficial spMohanty/PlantVillage-Dataset, pasta `raw/color`)

> Este documento foi gerado automaticamente por `step06_analysis_report.py` a partir dos arquivos de resultado dos Passos 1–5. Cada seção separa **fundamentação teórica**, **procedimento implementado**, **resultados obtidos** e **interpretação**. Explicações causais não testadas diretamente estão marcadas como *hipótese*. Revise e adapte a redação antes de entregar.


---
## 1. Análise e preparação do dataset

### 1.1 Fundamentação
**Vazamento de dados** (*data leakage*) ocorre quando informação do conjunto de teste influencia o treinamento, inflando as métricas (Kaufman et al., 2012). No PlantVillage isso é crítico: várias imagens são fotos diferentes (ou cópias aumentadas) **da mesma folha**; se uma ficar no treino e outra no teste, o classificador pode ‘reconhecer a folha’ em vez de aprender a doença. Por isso a divisão deve ser feita por **grupos** (folhas). O **desbalanceamento** das classes faz com que a acurácia seja dominada pelas classes grandes; por isso usamos também precisão, recall e F1 por classe e o **F1 macro** (média simples dos F1 por classe), que pesa todas as classes igualmente.

### 1.2 Procedimento implementado
1. Download por `git clone --filter=blob:none --sparse` + `sparse-checkout` de `raw/color` e `leaf_grouping` (somente o necessário). Nenhuma carga via Hugging Face foi usada.
2. Verificação da estrutura, das classes e da coerência com o README (nada foi presumido; divergências são reportadas).
3. Varredura paralela: MD5, decodificação completa (Pillow `verify`+`load`), dimensões.
4. Auditoria de rótulos: duplicatas exatas (MD5), fotos com mesmo nome em classes distintas, coerência com `leaf-map.json`.
5. Agrupamento por folha usando o `leaf-map.json` oficial (mesma regra de chave do loader oficial `plant_village.py`); para imagens sem mapeamento, blocos de fotos de numeração consecutiva (estratégia alternativa validada nos dados mapeados). Duplicatas exatas são unidas ao mesmo grupo (union-find).
6. Divisão `StratifiedGroupKFold(5)` (1 dobra = ~20% de teste), escolhendo entre 10 sementes a de proporções por classe mais próximas de 20% (critério usa apenas rótulos e grupos).

### 1.3 Resultados obtidos

- Pastas de classe em `raw/color`: **38**; arquivos: **54305** (README: 54.306 imagens; manifesto do repositório: color=54305, segmented=54306, grayscale=54305 — a diferença de 1 arquivo vem da própria pasta `segmented`).
- Culturas: **14**; pares cultura–doença (classes não saudáveis): **26**; nomes de doença distintos: 20 (README: 14 culturas e 26 doenças — coerente se contarmos pares cultura–doença).
- Imagens corrompidas: **0**. Dimensões: {'256x256': 54305}. Extensões: {'.jpg': 54303, '.png': 1, '.jpeg': 1}.
- Duplicatas exatas (MD5): **21** grupos (42 arquivos); entre classes distintas (conflito de rótulo): **0**. Foram mantidas 54284 imagens (removidas: {'duplicata exata (copia removida)': 21}).
- Nomes de foto repetidos em mais de uma classe: 2171 (sem MD5 idêntico: são colisões de numeração entre culturas, não rótulos conflitantes). 5783 arquivos pertencem a fotos com nome repetido (cópias aumentadas).
- Alias de rótulo do leaf-map detectado automaticamente: {'Apple_Frogeye Spot': 'Apple___Black_rot'} (mancha-olho-de-rã é a forma foliar da podridão negra da maçã; as imagens estão na pasta `Apple___Black_rot`).

**Balanceamento.** 38 classes; mín.=152, máx.=5507, mediana=1092; razão máx./mín.=36.2; maior classe: `Orange___Haunglongbing_(Citrus_greening)` (10.1% das imagens); entropia normalizada=0.919 (1 = perfeitamente balanceado). Um classificador que sempre prevê a classe majoritária acerta ≈ 10.1% (acurácia), mas com F1 macro muito baixo.

Figuras: `graficos/01_distribuicao_classes.png`, `01_distribuicao_culturas.png`, `01_exemplos_classes_1/2.png`. Tabela completa: `tabelas/01_classes_contagem.csv`.

**Investigação de divisões oficiais.** O GitHub não possui pasta `splits/` (verificado no manifesto: False); os arquivos de divisão citados no README_HF estão no Hugging Face (acessíveis nesta execução: False). O único candidato no repositório é `data_distribution_for_SVM/` (gerado para um experimento do artigo):

| candidato | n_treino | n_teste | cobertura_do_dataset | treino_teste_disjuntos | grupos_compartilhados_treino_teste | frac_teste |
|---|---|---|---|---|---|---|
| data_distribution_for_SVM (GitHub) | 8747 | 10544 | 0.355 | True | 132 | 0.547 |

Esse candidato cobre apenas parte do dataset, tem proporção invertida (mais teste que treino) e, sob o nosso agrupamento, compartilha grupos entre treino e teste; **foi rejeitado**. Adotamos a divisão própria agrupada.

**Validação da estratégia alternativa de grupos** (taxa de folhas reais — com `leaf_id` — que seriam partidas entre blocos de fotos consecutivas):

| largura_bloco | folhas_avaliadas | folhas_fragmentadas | taxa_fragmentacao |
|---|---|---|---|
| 16 | 7593 | 2414 | 0.318 |
| 32 | 7593 | 1225 | 0.161 |
| 64 | 7593 | 647 | 0.085 |
| 128 | 7593 | 345 | 0.045 |
| 256 | 7593 | 190 | 0.025 |

Escolhida a menor largura com fragmentação ≤ 5%: **128**. Tipos de grupo: {'leaf-map (oficial)': 41111, 'bloco de fotos consecutivas': 13191, 'imagem isolada': 3}.

**Divisão final:** treino=43363, teste=10921 (fração de teste=0.201); 7884 grupos; **grupos compartilhados entre treino e teste = 0**. Limitação: nas classes sem `leaf_id`, um simulado em imagens que possuem `leaf_id` indica que 4.8% das imagens de teste teriam a mesma folha no treino se só os blocos fossem usados; como só 2554 imagens de teste estão em grupos de bloco, a estimativa global do vazamento residual é ≈ **1.11%** do teste. As demais imagens usam o `leaf_id` oficial e têm vazamento de folha nulo.

### 1.4 Interpretação e dificuldades visuais
- **Fundo como atalho.** O fundo (cinza/arroxeado/marrom) varia conforme o local/dia de aquisição e se correlaciona com a cultura/classe. Um Random Forest treinado **apenas com a cor da borda da imagem** obteve 58.3% de acurácia no teste (acaso uniforme: 2.6%; classe majoritária: 2.6%). Isso mostra que parte do sinal ‘fácil’ do dataset não é a doença; resultados altos devem ser lidos com cautela (a borda de imagens de milho é folha, o que subestima o efeito). Figura: `01_dificuldade_fundo.png`.
- **Imagens muito parecidas entre doenças da mesma cultura** (ex.: manchas foliares), sombras, folhas que preenchem o quadro, variação de iluminação e várias fotos/cópias aumentadas por folha.
- **Desbalanceamento moderado/forte** (razão máx./mín. acima), exigindo métricas macro e atenção a `class_weight`.


---
## 2. Pré-processamento

### 2.1 Fundamentação
- **Redimensionamento** (INTER_AREA) reduz custo e uniformiza o vetor de características; reduzir demais perde textura fina.
- **Mediana/Gaussiano 3×3** atenuam ruído de sensor/JPEG, mas suavizam texturas finas.
- **CLAHE** (Zuiderveld, 1994) equaliza contraste localmente (aqui só no canal L do Lab); pode alterar a cor, que é discriminativa.
- **Espaços de cor:** HSV separa matiz/saturação de brilho (robustez a iluminação); Lab é perceptualmente mais uniforme. **Escala de cinza** é usada apenas para textura/gradientes (LBP, GLCM, HOG operam em intensidade).
- **Padronização** (z-score) é necessária para modelos baseados em distância/kernel (KNN, SVM), pois dimensões em escalas diferentes dominam a distância; árvores/florestas são invariantes a transformações monotônicas por característica.

### 2.2 Procedimento
Cada operação foi **avaliada** em 3482 imagens **somente de treino**, com validação cruzada agrupada (3 dobras) e Random Forest, nas configurações C1 (cor) e C2 (cor+textura). Regra de decisão: abandonar a linha de base (128 px, sem filtros) só se o ganho em F1 macro de C2 for ≥ 0,01.

### 2.3 Resultados

| variante | config | f1_macro_media | f1_macro_dp | acuracia_media |
|---|---|---|---|---|
| P0_128_base | C1_COR | 0.370 | 0.029 | 0.476 |
| P0_128_base | C2_COR+TEXTURA | 0.407 | 0.024 | 0.501 |
| P1_64 | C1_COR | 0.365 | 0.029 | 0.471 |
| P1_64 | C2_COR+TEXTURA | 0.420 | 0.024 | 0.509 |
| P2_256 | C1_COR | 0.379 | 0.029 | 0.482 |
| P2_256 | C2_COR+TEXTURA | 0.392 | 0.022 | 0.489 |
| P3_128_mediana3 | C1_COR | 0.379 | 0.032 | 0.474 |
| P3_128_mediana3 | C2_COR+TEXTURA | 0.433 | 0.024 | 0.521 |
| P4_128_gauss3 | C1_COR | 0.375 | 0.018 | 0.480 |
| P4_128_gauss3 | C2_COR+TEXTURA | 0.425 | 0.030 | 0.519 |
| P5_128_CLAHE | C1_COR | 0.371 | 0.016 | 0.484 |
| P5_128_CLAHE | C2_COR+TEXTURA | 0.404 | 0.032 | 0.510 |

Espaços de cor (variante base, RF):

| conjunto | n_features | f1_macro_media | f1_macro_dp |
|---|---|---|---|
| RGB | 30 | 0.293 | 0.022 |
| HSV | 30 | 0.359 | 0.020 |
| Lab | 6 | 0.263 | 0.007 |
| RGB+HSV+Lab | 66 | 0.370 | 0.029 |
| RGB+HSV+Lab+LBP+GLCM | 104 | 0.407 | 0.024 |

Efeito da padronização (config C2):

| modelo | tratamento | f1_macro_media | f1_macro_dp |
|---|---|---|---|
| KNN(k=7) | sem padronização | 0.203 | 0.015 |
| KNN(k=7) | com StandardScaler | 0.366 | 0.020 |
| SVM-RBF(C=10) | sem padronização | 0.182 | 0.011 |
| SVM-RBF(C=10) | com StandardScaler | 0.450 | 0.009 |
| RandomForest | sem padronização | 0.407 | 0.024 |
| RandomForest | com StandardScaler | 0.407 | 0.023 |

**Decisão:** `P3_128_mediana3` — parâmetros {'size': 128, 'denoise': 'median3', 'clahe': False}. Melhor variante na CV: `P3_128_mediana3` (ganho sobre a base: +0.026). Figuras: `02_exemplos_preprocessamento.png`, `02_histograma_clahe.png`, `02_comparacao_preprocessamento.png`.

### 2.4 Interpretação
- Em C2, a base obteve F1 macro 0.407±0.024 e a variante escolhida 0.433±0.024. O desvio-padrão entre dobras (≈0.026) é da ordem do ganho, logo a contribuição do pré-processamento é **pequena e de significância limitada** (amostra de ~3,5 mil imagens, 3 dobras). A decisão seguiu a regra de parcimônia definida antes do experimento.
- Ver acima o efeito dos espaços de cor: HSV/Lab agregam informação ao RGB? (compare `RGB` com `RGB+HSV+Lab`).
- A padronização muda drasticamente KNN e SVM e é irrelevante para a floresta, como previsto pela teoria.


---
## 3. Segmentação

### 3.1 Fundamentação
O fundo do PlantVillage é uma superfície quase neutra (croma baixo), enquanto a folha (verde, amarela, marrom) tem croma alto. No espaço **Lab**, o croma é √((a*−128)²+(b*−128)²). O **limiar de Otsu** (1979) escolhe o corte que maximiza a variância entre classes do histograma. **Morfologia** (abertura/fechamento) remove ruído e fecha lacunas; **contornos** externos preenchidos eliminam buracos internos (manchas de doença claras não podem virar ‘fundo’). Segmentar por ‘verde’ foi descartado a priori porque removeria as regiões doentes (marrons/amarelas), justamente as informativas.

### 3.2 Procedimento
Comparamos quatro métodos tradicionais (Otsu em S/HSV; k-means k=2 em Lab; Otsu no croma Lab com f=1; e o adotado, com fator f calibrado). Avaliação quantitativa por IoU/Dice contra as máscaras que acompanham o repositório (`raw/segmented`), **um proxy gerado pelos autores do dataset, não anotação manual**, usado só para calibrar f (metade A) e avaliar (metade B); nunca como característica.

### 3.3 Resultados

| metodo | n_imagens | iou_medio | iou_mediana | iou_p10 | dice_medio | frac_iou_abaixo_0.7 |
|---|---|---|---|---|---|---|
| Otsu em S (HSV) | 150 | 0.781 | 0.878 | 0.383 | 0.854 | 0.253 |
| k-means (k=2) em Lab | 150 | 0.661 | 0.756 | 0.279 | 0.760 | 0.480 |
| Otsu no croma Lab (f=1.0) | 150 | 0.857 | 0.918 | 0.638 | 0.913 | 0.140 |
| Otsu no croma Lab (f=0.7) [adotado] | 150 | 0.905 | 0.958 | 0.783 | 0.941 | 0.080 |

Fator do limiar escolhido: **0.7**. Limitações medidas em 1500 imagens de treino: máscaras inválidas=0.0%, folha ocupando >90% do quadro=4.5%, folha <15% do quadro=0.5%, área mediana=0.48. Figuras: `03_exemplos_segmentacao.png`, `03_melhores_e_piores_casos.png`, `03_distribuicao_iou.png`, `03_area_mascara.png`.

**Impacto na classificação (CV agrupada no treino):**

| config | variante | n_features | f1_macro_media | f1_macro_dp | acuracia_media |
|---|---|---|---|---|---|
| C1_COR | imagem inteira | 66 | 0.379 | 0.032 | 0.474 |
| C1_COR | segmentada | 66 | 0.325 | 0.022 | 0.434 |
| C2_COR+TEXTURA | imagem inteira | 104 | 0.433 | 0.024 | 0.521 |
| C2_COR+TEXTURA | segmentada | 104 | 0.380 | 0.023 | 0.493 |
| C3_COR+TEXTURA+GRADIENTES | imagem inteira | 443 | 0.423 | 0.021 | 0.529 |
| C3_COR+TEXTURA+GRADIENTES | segmentada | 443 | 0.408 | 0.015 | 0.513 |
| C4_C3+FORMA | segmentada | 457 | 0.436 | 0.011 | 0.526 |

### 3.4 Interpretação
- A máscara é boa (ver IoU), porém falha em folhas de baixo contraste cromático, sombras fortes e folhas que preenchem o quadro (milho); regiões escuras (necrose) próximas à borda podem ser cortadas.
- A decisão de usar ou não segmentação é **empírica** e é reavaliada no conjunto de teste (Seção 5).


---
## 4. Extração de características

### 4.1 Fundamentação
- **Cor:** média/desvio por canal (RGB, HSV, Lab) e histogramas descrevem clorose (amarelecimento), necrose (marrom) e manchas. Matiz é circular, por isso usamos a média vetorial (cos, sen).
- **Textura:** **LBP** uniforme (Ojala et al., 2002) codifica micropadrões locais (P=8,R=1 e P=16,R=2) de forma invariante à rotação; **GLCM** (Haralick et al., 1973) mede contraste, homogeneidade, energia, correlação e entropia (32 níveis, 4 ângulos promediados, distâncias 1 e 3) — úteis para pústulas, mofo e manchas.
- **Gradientes/bordas:** **HOG** (Dalal & Triggs, 2005) descreve a distribuição local de orientações; densidade de bordas (Canny), magnitude de gradiente (Sobel) e histograma global de orientações.
- **Forma** (da máscara): área, perímetro, circularidade 4πA/P², razões de aspecto, solidez, contato com a borda e 7 momentos de Hu (Hu, 1962, em escala log).

### 4.2 Procedimento
Extração em paralelo (joblib), em lotes de 2.000 imagens com cache (retomável); duas variantes por imagem lidas uma única vez. LBP é calculado uma vez por imagem e reutilizado. Na variante segmentada, estatísticas de cor/GLCM/LBP/gradientes são calculadas **apenas sobre pixels da máscara** (LBP e gradientes sobre a máscara erodida em 3 px); o HOG usa a imagem com fundo zerado.

### 4.3 Resultados
| grupo | n_colunas |
|---|---|
| hog | 324 |
| rgb | 30 |
| hsv | 30 |
| lbp | 28 |
| edge | 15 |
| shp | 14 |
| glcm | 10 |
| lab | 6 |

CSVs: `dados/features_imagem_completa.csv` e `dados/features_imagem_segmentada.csv` (colunas: `image_id`, `class`, `group_id`, `split` + características). **`image_id`, `class`, `group_id` e `split` nunca entram no classificador.** Descrição de cada coluna: `tabelas/04_descricao_features.csv`.

**Configurações comparadas:** C1 = cor; C2 = cor+textura; C3 = cor+textura+gradientes; C4 = C3 + forma (somente segmentada).

**Contribuição isolada de cada grupo** (RF, CV agrupada na subamostra de treino):
| grupo | n_features | f1_macro_cv | dp | acuracia_cv |
|---|---|---|---|---|
| HSV | 30 | 0.461 | 0.012 | 0.590 |
| RGB | 30 | 0.373 | 0.005 | 0.517 |
| LAB | 6 | 0.357 | 0.015 | 0.477 |
| HOG | 324 | 0.249 | 0.007 | 0.412 |
| LBP | 28 | 0.226 | 0.025 | 0.362 |
| GLCM | 10 | 0.209 | 0.001 | 0.308 |
| FORMA | 14 | 0.192 | 0.013 | 0.299 |
| BORDAS/GRADIENTES | 15 | 0.170 | 0.008 | 0.299 |

### 4.4 Interpretação
O grupo mais informativo isoladamente foi **HSV** e o menos informativo **BORDAS/GRADIENTES** (note que HOG tem 324 dimensões e os demais, poucas). Figuras: `04_pca_ilustrativa.png`, `04_heatmap_medias_por_classe.png`, `05_ablacao_grupos.png`.


---
## 5. Treinamento e avaliação

### 5.1 Fundamentação
- **KNN** classifica pelo voto dos k vizinhos mais próximos (distância euclidiana) — sensível à escala e à dimensionalidade; - **SVM com kernel RBF** maximiza a margem em um espaço de características induzido pelo kernel K(x,x′)=exp(−γ‖x−x′‖²) (Cortes & Vapnik, 1995); - **Random Forest** agrega árvores treinadas em amostras bootstrap com subconjuntos aleatórios de atributos (Breiman, 2001), reduz variância e não exige padronização.
Escolha: três famílias distintas (instâncias, margem/kernel, conjunto de árvores), todas tradicionais. **F1 macro** é adequado ao desbalanceamento; **DummyClassifier** estabelece o piso.

### 5.2 Procedimento
Pipeline `StandardScaler → classificador` (scaler ajustado só nos dados de treino de cada dobra); busca em grade (`GridSearchCV`, `StratifiedGroupKFold(3)`, F1 macro) em uma subamostra de ~6000 imagens de **treino**, a mesma para todas as tarefas (condições equivalentes); reajuste no treino completo; teste usado apenas na avaliação final. **Modo de execução: completo.** Grades efetivamente usadas — KNN: n_neighbors∈[3, 7, 15]; weights∈['uniform', 'distance']. Random Forest (200 árvores): max_features∈['sqrt', 0.3]; class_weight∈[None, 'balanced_subsample']. SVM (γ=‘scale’): C∈[1, 10, 100]; class_weight∈[None, 'balanced']. Random Forest não usa padronização (invariante a transformações monótonas por característica).

### 5.3 Resultados
**Referência (DummyClassifier):**
| referencia | acuracia_teste | f1_macro_teste | f1_ponderado_teste |
|---|---|---|---|
| DummyClassifier(most_frequent) | 0.1008 | 0.0048 | 0.0185 |
| DummyClassifier(stratified) | 0.0449 | 0.0249 | 0.0445 |

**Todas as combinações (ordenadas por F1 macro no teste):**
| variante | config | modelo | n_features | f1_macro_cv_treino | acuracia_teste | f1_macro_teste | precisao_macro_teste | recall_macro_teste | melhores_hiperparametros |
|---|---|---|---|---|---|---|---|---|---|
| inteira | C3_COR+TEXTURA+GRADIENTES | SVM | 443 | 0.627 | 0.949 | 0.932 | 0.938 | 0.928 | {"clf__C": 10, "clf__class_weight": "balanced"} |
| inteira | C2_COR+TEXTURA | SVM | 104 | 0.598 | 0.949 | 0.928 | 0.925 | 0.933 | {"clf__C": 10, "clf__class_weight": "balanced"} |
| segmentada | C4_C3+FORMA | SVM | 457 | 0.608 | 0.930 | 0.908 | 0.917 | 0.903 | {"clf__C": 10, "clf__class_weight": null} |
| segmentada | C3_COR+TEXTURA+GRADIENTES | SVM | 443 | 0.612 | 0.930 | 0.907 | 0.916 | 0.901 | {"clf__C": 10, "clf__class_weight": null} |
| inteira | C3_COR+TEXTURA+GRADIENTES | RandomForest | 443 | 0.570 | 0.916 | 0.892 | 0.922 | 0.873 | {"clf__class_weight": "balanced_subsample", "clf__max_features": "sqrt"} |
| inteira | C2_COR+TEXTURA | RandomForest | 104 | 0.562 | 0.915 | 0.887 | 0.900 | 0.877 | {"clf__class_weight": "balanced_subsample", "clf__max_features": 0.3} |
| segmentada | C4_C3+FORMA | RandomForest | 457 | 0.552 | 0.897 | 0.868 | 0.894 | 0.851 | {"clf__class_weight": "balanced_subsample", "clf__max_features": 0.3} |
| inteira | C1_COR | SVM | 66 | 0.520 | 0.904 | 0.867 | 0.859 | 0.880 | {"clf__C": 10, "clf__class_weight": "balanced"} |
| segmentada | C3_COR+TEXTURA+GRADIENTES | RandomForest | 443 | 0.538 | 0.887 | 0.856 | 0.889 | 0.835 | {"clf__class_weight": "balanced_subsample", "clf__max_features": 0.3} |
| inteira | C2_COR+TEXTURA | KNN | 104 | 0.542 | 0.893 | 0.852 | 0.858 | 0.852 | {"clf__n_neighbors": 3, "clf__weights": "distance"} |
| segmentada | C2_COR+TEXTURA | SVM | 104 | 0.530 | 0.883 | 0.846 | 0.847 | 0.850 | {"clf__C": 10, "clf__class_weight": "balanced"} |
| inteira | C1_COR | RandomForest | 66 | 0.525 | 0.887 | 0.843 | 0.864 | 0.832 | {"clf__class_weight": "balanced_subsample", "clf__max_features": 0.3} |
| segmentada | C2_COR+TEXTURA | RandomForest | 104 | 0.493 | 0.849 | 0.802 | 0.843 | 0.779 | {"clf__class_weight": "balanced_subsample", "clf__max_features": "sqrt"} |
| inteira | C3_COR+TEXTURA+GRADIENTES | KNN | 443 | 0.475 | 0.848 | 0.792 | 0.808 | 0.793 | {"clf__n_neighbors": 3, "clf__weights": "distance"} |
| segmentada | C4_C3+FORMA | KNN | 457 | 0.490 | 0.823 | 0.768 | 0.782 | 0.767 | {"clf__n_neighbors": 3, "clf__weights": "distance"} |
| inteira | C1_COR | KNN | 66 | 0.467 | 0.834 | 0.767 | 0.779 | 0.768 | {"clf__n_neighbors": 7, "clf__weights": "distance"} |
| segmentada | C3_COR+TEXTURA+GRADIENTES | KNN | 443 | 0.488 | 0.823 | 0.765 | 0.777 | 0.766 | {"clf__n_neighbors": 3, "clf__weights": "distance"} |
| segmentada | C1_COR | RandomForest | 66 | 0.423 | 0.801 | 0.739 | 0.776 | 0.721 | {"clf__class_weight": "balanced_subsample", "clf__max_features": 0.3} |
| segmentada | C1_COR | SVM | 66 | 0.412 | 0.784 | 0.718 | 0.707 | 0.746 | {"clf__C": 10, "clf__class_weight": "balanced"} |
| segmentada | C2_COR+TEXTURA | KNN | 104 | 0.444 | 0.785 | 0.711 | 0.728 | 0.712 | {"clf__n_neighbors": 7, "clf__weights": "distance"} |
| segmentada | C1_COR | KNN | 66 | 0.344 | 0.661 | 0.552 | 0.571 | 0.553 | {"clf__n_neighbors": 7, "clf__weights": "distance"} |

**Modelo escolhido por validação no treino:** `inteira__C3_COR+TEXTURA+GRADIENTES__SVM` (F1 macro CV=0.627); no teste: acurácia=0.949, F1 macro=0.932. Melhor no teste: `inteira__C3_COR+TEXTURA+GRADIENTES__SVM` (F1 macro=0.932).

Matrizes de confusão: `graficos/05_matriz_confusao_modelo_escolhido.png` e uma por algoritmo. Relatório de classificação completo: `relatorios/05_classification_report_modelo_escolhido.txt`. Métricas por classe: `tabelas/05_metricas_por_classe_modelo_escolhido.csv`.

**Classes com pior F1 (modelo escolhido):**
| classe | precisao | recall | f1 | suporte |
|---|---|---|---|---|
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.727 | 0.825 | 0.773 | 97 |
| Apple___Apple_scab | 0.850 | 0.816 | 0.833 | 125 |
| Tomato___Early_blight | 0.844 | 0.840 | 0.842 | 200 |
| Tomato___Late_blight | 0.863 | 0.842 | 0.852 | 380 |
| Tomato___Target_Spot | 0.891 | 0.824 | 0.856 | 267 |
| Potato___healthy | 0.929 | 0.812 | 0.867 | 32 |
| Raspberry___healthy | 1.000 | 0.784 | 0.879 | 74 |
| Corn_(maize)___Northern_Leaf_Blight | 0.955 | 0.858 | 0.904 | 246 |

**Pares de classes mais confundidos:**
| verdadeira | prevista | n_erros | pct_da_classe_verdadeira | mesma_cultura |
|---|---|---|---|---|
| Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite | 35 | 13.109 | True |
| Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 27 | 10.976 | True |
| Tomato___Late_blight | Tomato___Early_blight | 13 | 3.421 | True |
| Tomato___Early_blight | Tomato___Late_blight | 11 | 5.500 | True |
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | 11 | 3.274 | True |
| Grape___Black_rot | Grape___Esca_(Black_Measles) | 9 | 3.814 | True |
| Tomato___Early_blight | Tomato___Target_Spot | 8 | 4.000 | True |
| Tomato___Early_blight | Tomato___Bacterial_spot | 8 | 4.000 | True |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Bacterial_spot | 8 | 0.746 | True |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Orange___Haunglongbing_(Citrus_greening) | 8 | 0.746 | False |

Dos erros do modelo escolhido, **54.7%** ocorrem entre doenças da **mesma cultura**. Exemplos de erros: `05_erros_exemplos.png`.


---
## 6. Análise crítica dos resultados

**Configuração de características.** Média do F1 macro (teste, imagem inteira, 3 modelos): C1_COR=0.826; C2_COR+TEXTURA=0.889; C3_COR+TEXTURA+GRADIENTES=0.872. Passar de C1 a C2: +0.063; de C2 a C3: -0.017. Por modelo (teste, imagem inteira), C2−C1: KNN=+0.085; RandomForest=+0.044; SVM=+0.061; C3−C2: KNN=-0.060; RandomForest=+0.005; SVM=+0.004. Leitura: o sinal de cada delta indica se o grupo acrescentou informação útil àquele algoritmo. *Hipótese* para deltas negativos de C3−C2 em KNN: o HOG (324 de 443 colunas) domina a distância euclidiana e dilui cor/textura; SVM e Random Forest lidam melhor com muitas dimensões heterogêneas.

**Classificador.** Média do F1 macro por modelo (todas as configurações): SVM=0.872; RandomForest=0.841; KNN=0.744. Todos superam muito o DummyClassifier. As diferenças refletem a capacidade de cada família lidar com características heterogêneas e de alta dimensão.

**Segmentação.** Variação média do F1 macro de teste (segmentada − inteira) nos pares (config, modelo): **-0.096**; a segmentada ganhou em 0 de 9 pares. Adicionar forma (C4 − C3, segmentada), por modelo: KNN=+0.003; RandomForest=+0.012; SVM=+0.001. 
Portanto a segmentação **não** melhorou o desempenho. *Hipótese* (apoiada pelo teste do fundo da Seção 1.4, mas não isolada experimentalmente): o fundo carrega informação correlacionada à classe (local/dia de aquisição); ao removê-lo, perde-se um atalho que ajuda no conjunto de teste do próprio dataset, e erros de máscara (folhas cortadas, sombras) adicionam ruído. A segmentação é metodologicamente mais ‘honesta’ para generalização a campo, mesmo com métrica menor neste benchmark.

**Pré-processamento.** Ganho da variante escolhida sobre a base em C2 (CV, treino): +0.026 (desvio entre dobras ≈ 0.026); contribuição pequena e de significância limitada; o fator decisivo foi a **padronização** para KNN/SVM (Seção 2.3).

**Quais classes foram mais confundidas e por quê.** Ver pares acima. Os pares mais confundidos são de doenças da mesma cultura com sintomas parecidos, mas apenas 54.7% de todos os erros são intra-cultura: o restante cruza culturas. *Hipótese* (não isolada): lesões pequenas não são capturadas por características globais a 128 px, e o fundo/iluminação de cada sessão de aquisição contribui para erros entre culturas.

**Limitações que permanecem.**
- Vazamento residual estimado ≈ 1.11% do teste (classes sem `leaf_id`);
- fundo como atalho e imagens de bancada (laboratório): o desempenho não representa fotos reais em campo;
- a seleção de hiperparâmetros usa subamostra (por custo) e um grid pequeno;
- HOG não é invariante à rotação; características globais não localizam lesões;
- máscaras avaliadas contra um proxy (não manual);
- um único split de teste (sem intervalo de confiança).

**Melhorias futuras (dentro das restrições do enunciado).** Características por regiões da folha (grade) ou da lesão segmentada (limiarização no matiz/saturação dentro da máscara), histogramas de cor em mais espaços/bins, LBP rotacional multiescala, seleção de características/PCA ajustados no treino, busca de hiperparâmetros mais ampla com a subamostra maior, validação com vários splits e intervalos de confiança, e avaliação com remoção explícita do fundo (preenchimento neutro) para quantificar o atalho.


---
## Referências
- Hughes, D. P.; Salathé, M. *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060, 2015.
- Mohanty, S. P.; Hughes, D. P.; Salathé, M. *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science, 7:1419, 2016 (origem do dataset e do agrupamento por folha; não usamos deep learning).
- Kaufman, S.; Rosset, S.; Perlich, C.; Stitelman, O. *Leakage in data mining: formulation, detection, and avoidance*. ACM TKDD, 6(4), 2012.
- Otsu, N. *A threshold selection method from gray-level histograms*. IEEE Trans. Systems, Man, and Cybernetics, 9(1), 1979.
- Zuiderveld, K. *Contrast limited adaptive histogram equalization*. Graphics Gems IV, Academic Press, 1994.
- Ojala, T.; Pietikäinen, M.; Mäenpää, T. *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI, 24(7), 2002.
- Haralick, R. M.; Shanmugam, K.; Dinstein, I. *Textural features for image classification*. IEEE Trans. Systems, Man, and Cybernetics, 3(6), 1973.
- Dalal, N.; Triggs, B. *Histograms of oriented gradients for human detection*. IEEE CVPR, 2005.
- Hu, M.-K. *Visual pattern recognition by moment invariants*. IRE Trans. Information Theory, 8(2), 1962.
- Cover, T.; Hart, P. *Nearest neighbor pattern classification*. IEEE Trans. Information Theory, 13(1), 1967.
- Cortes, C.; Vapnik, V. *Support-vector networks*. Machine Learning, 20, 1995.
- Breiman, L. *Random forests*. Machine Learning, 45(1), 2001.
- Pedregosa, F. et al. *Scikit-learn: machine learning in Python*. JMLR, 12, 2011. Documentação: https://scikit-learn.org/stable/
- van der Walt, S. et al. *scikit-image: image processing in Python*. PeerJ, 2:e453, 2014. Documentação: https://scikit-image.org/docs/stable/
- Bradski, G. *The OpenCV Library*. Dr. Dobb’s Journal, 2000. Documentação: https://docs.opencv.org/
- Repositório do dataset: https://github.com/spMohanty/PlantVillage-Dataset
