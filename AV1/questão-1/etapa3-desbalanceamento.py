"""
Etapa 3 - Tratamento do desbalanceamento (SOMENTE no treino).

Gera:
  questão-1/dados_processados/class_weights.json         pesos por classe (calculados no treino)
  questão-1/dados_processados/train_oversampled.parquet  treino com classes minoritárias duplicadas
validation e test NÃO são tocados: precisam refletir a distribuição real.
"""
import json
import pandas as pd
from preprocessamento import PASTA_SAIDA, carregar_processado

SEMENTE = 42  # fixa o sorteio: resultado reproduzível
ROTULOS = {0: "NONE", 1: "ENTAILMENT", 2: "PARAPHRASE"}

treino = carregar_processado("train_processado.parquet")
contagem = treino["entailment_judgment"].value_counts().sort_index()
n_total, n_classes = len(treino), len(contagem)

# peso = n_total / (n_classes * n_da_classe)  -> mesma fórmula do 'balanced' do scikit-learn
pesos = {int(c): n_total / (n_classes * int(n)) for c, n in contagem.items()}
with open(PASTA_SAIDA / "class_weights.json", "w", encoding="utf-8") as f:
    json.dump(pesos, f, indent=2)

print("Classe | rótulo      | exemplos | peso")
for c, n in contagem.items():
    print(f"{c:6d} | {ROTULOS[c]:11s} | {n:8d} | {pesos[int(c)]:.3f}")

# Oversampling aleatório: sorteia COM reposição até igualar a classe majoritária
alvo = int(contagem.max())
partes = []
for c in contagem.index:
    parte = treino[treino["entailment_judgment"] == c]
    if len(parte) < alvo:
        extra = parte.sample(n=alvo - len(parte), replace=True, random_state=SEMENTE)
        parte = pd.concat([parte, extra])
    partes.append(parte)
balanceado = (pd.concat(partes)
              .sample(frac=1, random_state=SEMENTE)   # embaralha
              .reset_index(drop=True))
balanceado.to_parquet(PASTA_SAIDA / "train_oversampled.parquet", index=False)

print("\nTreino após oversampling:")
print(balanceado["entailment_judgment"].value_counts().sort_index().to_string())