## AV2 - Classificação de imagens com extração de características

## Descrição Geral

Cada equipe receberá um dataset de imagens e deverá desenvolver uma solução para classificar as imagens nas categorias indicadas. A solução deverá utilizar características extraídas das imagens e algoritmos tradicionais de aprendizado de máquina.

O objetivo é conduzir um mini-projeto completo de PDI, incluindo etapas de pré-processamento, análise exploratória, classificação e avaliação quantitativa dos resultados.

As respostas devem ser apresentadas no próprio repositório, em um arquivo, contendo descrição metodológica, gráficos, tabelas e explicações objetivas.

Não será permitido utilizar redes neurais convolucionais, técnicas de aprendizado profundo ou modelos pré-treinados para extração de características, segmentação ou classificação.

### 1. Análise e preparação do dataset
A equipe deverá conhecer e organizar os dados antes de iniciar a modelagem. Essa etapa deverá contemplar:
- Identificação das classes e da quantidade de imagens por classe.
- Verificação de imagens corrompidas, duplicadas ou com problemas de rotulagem.
- Análise do balanceamento entre as classes.
- Apresentação de exemplos das imagens e das principais dificuldades do problema.
- Divisão dos dados em conjuntos de treinamento e teste, com justificativa da estratégia utilizada.
- Imagens do mesmo objeto, indivíduo ou sequência, quando existentes, deverão permanecer no mesmo conjunto para evitar vazamento de dados.

### 2. Pré-processamento das imagens
A equipe deverá selecionar e avaliar operações de pré-processamento adequadas ao problema, justificando suas escolhas.

Poderão ser consideradas operações como redimensionamento, conversão de espaço de cores, redução de ruído, ajuste de contraste e normalização.

Não é necessário aplicar todas essas operações. Cada técnica utilizada deverá ter uma finalidade relacionada às características do dataset.

### 3. Segmentação, quando necessária

A equipe deverá avaliar se a separação do objeto de interesse e do fundo pode contribuir para a classificação.

Caso utilize segmentação, deverá apresentar o método empregado, exemplos dos resultados e suas limitações. Poderão ser utilizados métodos tradicionais, como limiarização, operações morfológicas, detecção de contornos e agrupamento de pixels.

Caso não utilize segmentação, deverá justificar essa decisão com base nas imagens e no objetivo da classificação.

### 4. Extração de características e construção do dataframe

A equipe deverá transformar cada imagem em um vetor numérico de características. Poderão ser exploradas características de:
- Cor: médias, desvios e histogramas dos canais.
- Textura: LBP e medidas derivadas da matriz de coocorrência.
- Forma: área, perímetro, circularidade, proporções e momentos de Hu.
- Gradientes e bordas: HOG, densidade de bordas e orientação dos gradientes.
A escolha deverá ser justificada em função do problema. A equipe deverá comparar pelo menos duas configurações de características, por exemplo, cor isoladamente e cor combinada com textura.

O dataframe final deverá conter uma linha por imagem, colunas com as características extraídas, uma coluna com a classe e uma coluna com o identificador da imagem. O identificador não deverá ser utilizado como entrada do classificador.

### 5. Treinamento e avaliação dos classificadores

A equipe deverá comparar pelo menos dois algoritmos tradicionais de classificação, como KNN, SVM, árvore de decisão, Random Forest ou regressão logística.

A avaliação deverá contemplar:
- Uma referência simples de desempenho, como um classificador que sempre prevê a classe mais frequente.
- Seleção de configurações e hiperparâmetros utilizando somente os dados de treinamento, por validação cruzada ou conjunto de validação.
- Padronização das características quando adequada ao algoritmo.
- Acurácia, precisão, recall e F1-score por classe, além do F1 macro e da matriz de confusão.
- Análise de exemplos classificados incorretamente.
O conjunto de teste deverá ser reservado para a avaliação final. Procedimentos que aprendem parâmetros dos dados, como padronização, seleção de características e PCA, deverão ser ajustados somente nos dados de treinamento de cada divisão.

### 6. Análise dos resultados
A equipe deverá discutir quais características e classificadores apresentaram melhores resultados, quais classes foram mais confundidas e quais fatores podem explicar os erros.

A discussão também deverá avaliar a contribuição do pré-processamento e, quando utilizada, da segmentação. Não basta apresentar o melhor resultado: é necessário demonstrar como as decisões do pipeline influenciaram o desempenho.


# Critérios de Avaliação

* Clareza metodológica e organização do relatório
* Qualidade do código e estrutura do projeto
* Profundidade na análise exploratória
* Correção e justificativa das escolhas técnicas
* Qualidade da avaliação quantitativa
* Capacidade de reflexão crítica

---

##  Aluno/dupla - Dataset


Abimael e Kaue: https://github.com/spMohanty/PlantVillage-Dataset

Calina e Jefter: https://github.com/garythung/trashnet

Dayanne , Hellen, Suiany: https://github.com/fruits-360/fruits-360-100x100

Elyson e Ramon: https://isy.gitlab-pages.liu.se/cvl/en/datasets/swedish-leaf/

Matias- Kelsey - Kevin: https://data.lhncbc.nlm.nih.gov/public/Malaria/index.html

### Observações para o Relatório 

Discutir **organizadamente** os resultados obtidos de cada questão.
Ao concluir o relatório, compartilhar na pasta AV2 da sua brach no github


### Observações para o Apresentação

Criar apresentação para realizar até 26/11
