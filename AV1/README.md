# NLP Buscapé - Classificação e Análise de Sentimentos

Pipeline completo de Processamento de Linguagem Natural (NLP) em Python 3.11, desenvolvido sob os princípios de **Programação Orientada a Objetos (POO)** e boas práticas de arquitetura modular, aplicado ao dataset de avaliações de e-commerce do Buscapé.

---

## 📁 Estrutura do Repositório

```text
AV1/
├── pre_process_and_analisis/        # PARTES 1 e 2: EDA, Pré-processamento e Aumento de Dados
│   ├── cache_checkpoint/            # Cache incremental para multiprocessamento (.pkl)
│   ├── data/                        # Datasets gerados (Preprocessados, SMOTE e EDA)
│   ├── visualizations/              # Gráficos gerados (EDA, Nuvens de Palavras e Clusters)
│   ├── utils/                       # Dicionários customizados e mapas de abreviações
│   ├── text_preprocessor.py         # Orquestrador principal do pré-processamento paralelo
│   ├── text_cleaner.py              # Limpeza e normalização de strings
│   ├── spell_checker.py             # Correção ortográfica em dois passes com SymSpell + Expansão de Gírias
│   ├── lemmatizer.py                # Tokenização, remoção de stopwords e lematização via spaCy
│   ├── smote_handler.py             # Balanceamento sintético otimizado anti-estouro de RAM
│   ├── eda_augmenter.py             # Aumento de dados por Easy Data Augmentation (EDA)
│   ├── corpus_analyzer.py           # Análise descritiva, estatísticas, K-Means e LDA
│   └── analise_and_preprocess.py    # Orquestrador central da Parte 1 e 2
│
├── classification/                  # PARTE 3: Modelagem e Aprendizado
│   ├── text_classifier.py           # Encapsulamento dos 4 modelos base (Naive Bayes, SVM, LogReg, MLP)
│   └── model_trainer.py             # Treinamento por época (Loss/Acurácia) e curvas de aprendizado
│
├── evaluation/                      # PARTE 4: Avaliação Consolidada
│   ├── model_evaluator.py           # Métricas quantitativas (F1-score, Acurácia, Precision, Recall)
│   └── results/                     # Gráficos de curvas de aprendizado e relatórios de avaliação
│
├── main.py                          # Entrypoint central orquestrando o pipeline completo
├── buscape.csv                      # Dataset original brutó (Necessário na raiz)
├── requirements.txt                 # Dependências do projeto (Python 3.11)
└── README.md                        # Documentação do projeto

```

---

## 🔄 Fluxo de Execução do Pipeline

O pipeline opera de forma integrada e sequencial quando o comando principal é acionado:

### 1. Ingestão e Análise Exploratória Inicial (EDA)

* O `main.py` carrega o arquivo `buscape.csv` e valida a presença das colunas de texto e rótulo.
* O `CorpusExploratoryAnalyzer` calcula as métricas descritivas do corpus bruto (tamanho total, média de palavras/caracteres, distribuição de classes) e exporta os relatórios para a pasta de visualizações.

### 2. Pré-processamento Paralelizado e Limpeza Textual

Gerenciado pelo `TextPreprocessor` utilizando múltiplos núcleos de CPU com checkpoints incrementais em cache:

* **Limpeza (`text_cleaner.py`):** Remoção de URLs, menções, caracteres especiais e normalização de espaços.
* **Correção Ortográfica (`spell_checker.py`):** Utiliza o SymSpell em dois passes. No **Passe 1**, converte automaticamente abreviações e gírias da internet e e-commerce (ex: *pq -> por que*, *obg -> obrigado*, *vc -> voce*) mapeadas no `ABBREVIATIONS_MAP` e aplica correção de distância editável com preservação de termos críticos do domínio. O **Passe 2** realiza um refinamento focado em palavras raras.
* **Lematização (`lemmatizer.py`):** Executado via `spaCy` (`pt_core_news_sm`), extrai o lema real de cada palavra e remove *stopwords* irrelevantes, preservando termos de negação essenciais (*não, nunca, jamais*).

### 3. Aumento de Dados Otimizado (SMOTE & EDA)

Para mitigar o forte desbalanceamento de classes do corpus original:

* **SMOTE (`smote_handler.py`):** Converte o texto pré-processado em matrizes esparsas via TF-IDF restrito e aplica sobreamostragem sintética de forma controlada, com travas de segurança contra estouro de memória RAM e fallback robusto.
* **EDA (`eda_augmenter.py`):** Aplica técnicas de substituição e inserção sinônima baseada em regras no corpus da classe minoritária.

### 4. Análise Não Supervisionada (K-Means & LDA)

* O corpus limpo passa por vetorização TF-IDF e CountVectorizer para o agrupamento não supervisionado.
* O K-Means e a Latent Dirichlet Allocation (LDA) agrupam os documentos em tópicos representativos, rotulam os grupos com base nas palavras mais frequentes, geram gráficos de distribuição e exportam os resultados mapeados em CSV.

### 5. Treinamento Comparativo e Curvas de Aprendizado

Gerenciado pelo `ModelTrainer` utilizando os 4 classificadores do projeto (`Naive Bayes`, `SVM`, `Regressão Logística` e `MLP`):

* Os modelos iterativos (`MLP` e `Regressão Logística` / `SGD`) são treinados época por época utilizando estratégias incrementais (`warm_start`), calculando e plotando a evolução da **Loss** e da **Acurácia** (treino vs teste) ao longo das épocas.
* Os modelos não iterativos (`Naive Bayes` e `SVM`) geram curvas de aprendizado baseadas na variação do tamanho de sub-amostras do dataset.

### 6. Avaliação Consolidada

* O `ModelEvaluator` consolida as predições de todos os experimentos (SMOTE vs EDA) gerando relatórios quantitativos definitivos (Acurácia, F1-Score, Precision, Recall) e elegendo o melhor modelo global.

---

## 🚀 Setup e Execução

### 1. Criar e Ativar o Ambiente Virtual (`venv`)

* **Windows** (via Python Launcher `py`):
```cmd
py -3.11 -m venv venv
venv\Scripts\activate

```


* **Linux / macOS**:
```bash
python3.11 -m venv venv
source venv/bin/activate

```



### 2. Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download pt_core_news_sm

```

### 3. Posicionar o Dataset

Adicione o ficheiro do dataset na raiz do projeto: `AV1/buscape.csv`.

### 4. Executar o Pipeline Completo

```bash
python main.py

```