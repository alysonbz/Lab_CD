
from pathlib import Path
import pandas as pd

# Localizar a pasta do projeto
pasta_projeto = Path(__file__).resolve().parent

# Localizar a pasta do dataset
pasta_dataset = pasta_projeto / "dataset"

# Carregar o arquivo de treinamento
df_treino = pd.read_parquet(
    pasta_dataset / "train-00000-of-00001.parquet"
)

# Exibir as primeiras linhas
print("Primeiras linhas do dataset:")
print(df_treino.head())

# Exibir as colunas
print("\nColunas disponíveis:")
print(df_treino.columns.tolist())

# Exibir o tamanho do dataset
print("\nQuantidade de linhas e colunas:")
print(df_treino.shape)

# Exibir a distribuição das classes
print("\nDistribuição das classes:")
print(df_treino["entailment_judgment"].value_counts())

print("\nExemplos de cada classe:")

for classe in sorted(df_treino["entailment_judgment"].unique()):
    print(f"\nClasse {classe}:")

    exemplo = df_treino[
        df_treino["entailment_judgment"] == classe
        ].iloc[0]

    print("Premissa:", exemplo["premise"])
    print("Hipótese:", exemplo["hypothesis"])
    print("Pontuação de similaridade:", exemplo["relatedness_score"])
