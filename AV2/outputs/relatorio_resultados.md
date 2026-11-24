# Relatório de classificação de resíduos

## 1. Metodologia

Foram utilizadas características tradicionais de processamento digital de imagens, sem redes neurais ou modelos pré-treinados.

Foram comparadas três configurações: características de cor, cor combinada com textura e configuração completa, incluindo forma e gradientes.

O balanceamento foi realizado exclusivamente no conjunto de treinamento por Data Augmentation.

## 2. Comparação dos modelos

| configuracao   | modelo        |   acuracia |   precisao_macro |   recall_macro |   f1_macro |
|:---------------|:--------------|-----------:|-----------------:|---------------:|-----------:|
| cor_textura    | SVM           |   0.809901 |         0.797677 |       0.808414 |   0.801245 |
| cor            | Random Forest |   0.782178 |         0.75743  |       0.78208  |   0.764746 |
| completa       | SVM           |   0.754455 |         0.757478 |       0.746355 |   0.750358 |

## 3. Melhor resultado

**Classificador:** SVM

**Características:** cor_textura

**Acurácia:** 0.8099

**F1 Macro:** 0.8012

## 4. Desempenho por classe

| classe    |   precisao |   recall |   f1_score |   suporte |
|:----------|-----------:|---------:|-----------:|----------:|
| trash     |   0.677419 | 0.777778 |   0.724138 |        27 |
| metal     |   0.759036 | 0.768293 |   0.763636 |        82 |
| glass     |   0.72807  | 0.83     |   0.775701 |       100 |
| plastic   |   0.853933 | 0.791667 |   0.821622 |        96 |
| paper     |   0.877358 | 0.781513 |   0.826667 |       119 |
| cardboard |   0.890244 | 0.901235 |   0.895706 |        81 |

## 5. Análise dos erros

Foram identificados 96 erros no conjunto de teste.

- metal confundido com glass: 10 ocorrência(s).

- plastic confundido com glass: 10 ocorrência(s).

- paper confundido com metal: 9 ocorrência(s).

- paper confundido com glass: 8 ocorrência(s).

- glass confundido com metal: 6 ocorrência(s).

- paper confundido com plastic: 6 ocorrência(s).

- plastic confundido com paper: 4 ocorrência(s).

- plastic confundido com metal: 3 ocorrência(s).

- metal confundido com paper: 3 ocorrência(s).

- glass confundido com trash: 3 ocorrência(s).


## 6. Discussão

As características de cor podem ajudar a distinguir materiais com propriedades visuais diferentes. Entretanto, vidro, plástico e metal podem apresentar reflexos e cores semelhantes dependendo da iluminação.

As características de textura podem auxiliar na distinção de superfícies lisas, rugosas, amassadas ou com padrões impressos.

As características de forma podem contribuir na identificação de garrafas, caixas e embalagens. Entretanto, sua confiabilidade depende da qualidade da segmentação do objeto.

A utilização de Data Augmentation aumenta a diversidade das imagens de treinamento, mas não substitui a coleta de novos exemplos reais.

Como o dataset não disponibiliza identificadores de objetos físicos, não é possível garantir que fotografias do mesmo objeto estejam em partições diferentes. Essa limitação deve ser considerada na interpretação dos resultados.


## 7. Conclusão

A melhor configuração experimental foi cor_textura, utilizando SVM, com F1 Macro de 0.8012. A escolha deve ser interpretada considerando as limitações do dataset e a análise das classes com maior índice de confusão.
