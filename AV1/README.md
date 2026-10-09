# NLP Buscapé - Classificação e Análise de Sentimentos

Este projeto é um pipeline completo de Processamento de Linguagem Natural (NLP) em Python 3.11, desenvolvido sob os princípios de **Programação Orientada a Objetos (POO)** e boas práticas de arquitetura de software.

O projeto abrange as 5 etapas da atividade de NLP sobre o dataset de avaliações do Buscapé:

1. **Pré-processamento e Limpeza Textual** *(Implementado)*
2. **Análise Exploratória do Corpus** *(Implementado)*
3. **Modelos de Classificação de Texto** *(Esboço / Em expansão)*
4. **Avaliação Quantitativa dos Resultados** *(Esboço / Em expansão)*
5. **Relatório e Conclusões** *(Esboço / Em expansão)*

---

## 📁 Estrutura do Projeto

```text
AV1/
├── pre_process_and_analisis/      # PARTES 1 e 2 (Implementadas)
│   ├── text_preprocessor.py       # Limpeza, stemming, TF-IDF / CountVectorizer e SMOTE
│   ├── corpus_analyzer.py         # EDA, estatísticas, gráficos e nuvem de palavras
│   └── analise_and_preprocess.py  # Orquestrador das Partes 1 e 2
│
├── classification/                # PARTE 3 (Esboço / A Adicionar)
│   ├── text_classifier.py         # Encapsulamento dos modelos (Naive Bayes, SVM, Logistic Regression, MLP)
│   └── model_trainer.py           # Pipeline de treino, validação e cross-validation
│
├── evaluation/                    # PARTE 4 (Esboço / A Adicionar)
│   └── model_evaluator.py         # Cálculo de métricas (Accuracy, F1, Precision, Recall) e matrizes de confusão
│
├── main.py                        # Entrypoint central do pipeline completo
├── requirements.txt               # Dependências do projeto (Python 3.11)
└── README.md                      # Documentação do projeto
```

---

## 🚀 Setup e Execução

### 1. Criar e Ativar o Ambiente Virtual (`venv`)

Para garantir o uso do **Python 3.11**, utilize o comando apropriado para o seu sistema operacional:

* **Windows** (usando o Python Launcher `py`):
  ```cmd
  py -3.11 -m venv venv
  venv\Scripts\activate
  ```

* **Linux**:
  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  ```

* **macOS** (via Homebrew ou instalador oficial):
  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  ```

### 2. Instalar Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Posicionar o Dataset
Adicione o ficheiro `buscape.csv` baixado do Kaggle na raiz do projeto (`AV1/buscape.csv`).

> ⚠️ **Nota:** A presença do ficheiro `buscape.csv` na raiz é obrigatória. Se não for encontrado, a aplicação lança um erro (`FileNotFoundError`) e interrompe a execução.

### 4. Executar o Pipeline
```bash
python main.py
```

---

## 🛠️ Esboço da Adição das Próximas Partes

As novas etapas foram arquitetadas para serem integradas de forma modular sem alterar o código existente nas Partes 1 e 2:

### 🔹 Parte 3: Construção dos Modelos (`classification/`)
* **`text_classifier.py`**: Classe responsável por inicializar e configurar os classificadores (ex.: `MultinomialNB`, `LogisticRegression`, `SVC`, `MLPClassifier`).
* **`model_trainer.py`**: Classe responsável por dividir os dados (treino/teste), treinar os modelos comparando **CountVectorizer** vs **TF-IDF**, e aplicar validação cruzada.

### 🔹 Parte 4: Avaliação Quantitativa (`evaluation/`)
* **`model_evaluator.py`**: Classe responsável por gerar tabelas comparativas com métricas de desempenho (Acurácia, Precision, Recall, F1-Score) e plotar as matrizes de confusão.

### 🔹 Parte 5: Conclusões e Relatório
* Apresentação das respostas reflexivas sobre limitações, dificuldades encontradas, propostas de melhoria e aplicações no mundo real.

---

## 🤝 Como Contribuir e Adicionar Novas Partes

1. Crie o módulo correspondente na pasta indicada (ex.: `classification/text_classifier.py`).
2. Siga os princípios de POO (encapsulamento e responsabilidade única).
3. Importe e chame os novos módulos no `main.py` após o término da execução do `run_pipeline()`.
4. Abra um Pull Request com a nova funcionalidade implementada.
