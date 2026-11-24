# Classificação de Resíduos por Processamento Digital de Imagens

## 1. Sobre o projeto

Este repositório apresenta o desenvolvimento de uma solução de classificação automática de resíduos utilizando técnicas tradicionais de Processamento Digital de Imagens (PDI) e Aprendizado de Máquina.

O projeto foi desenvolvido como parte da avaliação AV2 e tem como objetivo classificar imagens de resíduos em seis categorias:

- **Cardboard:** papelão.
- **Glass:** vidro.
- **Metal:** metal.
- **Paper:** papel.
- **Plastic:** plástico.
- **Trash:** outros resíduos.

A solução utiliza características visuais extraídas diretamente das imagens, como cor, textura, forma e gradientes, para treinar e avaliar diferentes algoritmos tradicionais de classificação.

### Objetivos

- Realizar análise exploratória e preparação do dataset.
- Investigar problemas de qualidade e balanceamento das imagens.
- Aplicar técnicas de pré-processamento adequadas às características dos resíduos.
- Extrair características visuais e construir representações numéricas das imagens.
- Comparar diferentes algoritmos tradicionais de aprendizado de máquina.
- Avaliar quantitativamente os resultados e identificar as principais dificuldades de classificação.

**Restrição metodológica:** não são utilizadas redes neurais convolucionais (CNNs), técnicas de Deep Learning ou modelos pré-treinados. Toda a extração de características e classificação é realizada por métodos tradicionais.

---

## 2. Dataset utilizado

O projeto utiliza o dataset **TrashNet**, composto por imagens de resíduos distribuídas em seis categorias.

As imagens apresentam, predominantemente, um único objeto de interesse em foco, como garrafas, caixas de papelão, recipientes de vidro, latas e embalagens.

Essa característica permite investigar atributos visuais relacionados aos materiais e às propriedades geométricas dos objetos.

### Distribuição das classes

A distribuição original observada no dataset apresenta diferenças significativas entre as categorias.

A classe `paper` possui maior quantidade de imagens, enquanto `trash` apresenta uma quantidade consideravelmente menor.

Esse desbalanceamento pode prejudicar o desempenho dos classificadores, principalmente em relação às classes minoritárias.

Por esse motivo, o projeto utiliza Data Augmentation exclusivamente no conjunto de treinamento.

---

## 3. Estrutura do projeto

```text
projeto_analise_dataset/
│
├── main.py
├── requirements.txt
├── README.md
│
├── analise_dataset/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── exploracao.py
│   ├── limpeza.py
│   ├── divisao.py
│   ├── preprocessamento.py
│   ├── segmentacao.py
│   ├── features.py
│   ├── treinamento.py
│   └── analise_resultados.py
│
├── data/
│   ├── dataset-resized/
│   ├── dataset_limpo/
│   └── dataset_aumentado/
│
└── outputs/
    ├── split.json
    ├── dataframe_cor.csv
    ├── dataframe_cor_textura.csv
    ├── dataframe_completo.csv
    ├── metricas_modelos.csv
    ├── metricas_validacao.csv
    ├── matriz_confusao_*.png
    ├── comparacao_modelos.png
    ├── comparacao_acuracia.png
    ├── erros_*.csv
    └── relatorio_resultados.md
```

### Responsabilidade dos módulos

| Arquivo | Descrição |
|---|---|
| `main.py` | Ponto de entrada e execução das etapas. |
| `config.py` | Configurações, caminhos e classes. |
| `dataset.py` | Download e extração do dataset. |
| `exploracao.py` | Análise exploratória e distribuição das classes. |
| `limpeza.py` | Identificação e remoção de imagens problemáticas. |
| `divisao.py` | Divisão estratificada entre treinamento e teste. |
| `preprocessamento.py` | Redimensionamento, redução de ruído e ajuste de contraste. |
| `segmentacao.py` | Segmentação tradicional por limiarização de Otsu. |
| `features.py` | Extração e organização das características. |
| `treinamento.py` | Balanceamento, treinamento e avaliação dos classificadores. |
| `analise_resultados.py` | Comparação dos resultados e geração do relatório. |

---

# 4. Metodologia

O desenvolvimento foi organizado em seis etapas, seguindo as orientações da avaliação.

## 4.1. Análise e preparação do dataset

Inicialmente, é realizada a exploração do conjunto de dados para identificar as classes existentes e suas respectivas quantidades de imagens.

Também são realizadas verificações de integridade para identificar arquivos corrompidos e duplicatas exatas.

Após essa análise, é criada uma versão limpa do dataset.

A divisão dos dados é realizada de maneira estratificada, preservando aproximadamente a proporção das classes nos conjuntos de treinamento e teste.

A proporção utilizada é:

- 80% para treinamento.
- 20% para teste.

A divisão utiliza uma semente aleatória fixa (`random_state=42`), permitindo reproduzir os experimentos.

**Limitação:** o TrashNet não disponibiliza identificadores de objetos físicos ou sessões de captura. Portanto, não é possível garantir completamente que fotografias do mesmo objeto físico permaneçam em uma única partição.

## 4.2. Pré-processamento das imagens

O pré-processamento busca reduzir variações visuais que possam prejudicar a extração de características.

São utilizadas as seguintes operações:

**Redimensionamento e padronização**

As imagens são redimensionadas para uma resolução de 128 × 128 pixels, preservando a proporção original do objeto.

Quando necessário, são adicionadas margens para completar as dimensões.

**Redução de ruído**

É aplicado o filtro bilateral, que busca suavizar pequenas variações e preservar informações de bordas.

**Ajuste de contraste**

É utilizado o CLAHE (Contrast Limited Adaptive Histogram Equalization), aplicado ao canal de luminância do espaço de cores LAB.

Essa operação procura melhorar a visibilidade de detalhes sem alterar diretamente os