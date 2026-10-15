"""
Etapa 2
"""
from preprocessamento import (PASTA_SAIDA, SPLITS, carregar_nlp,
                              carregar_split, preprocessar_dataframe)

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
nlp = carregar_nlp()

for split in SPLITS:
    df = carregar_split(split)
    processado = preprocessar_dataframe(df, nlp)
    destino = PASTA_SAIDA / f"{split}_processado.parquet"
    processado.to_parquet(destino, index=False)
    print(f"{split:10s} {len(processado):5d} linhas -> {destino.name}")

print("\nExemplo (treino, linha 0):")
ex = preprocessar_dataframe(carregar_split("train").head(1), nlp).iloc[0]
for col in ["premise", "premise_norm", "premise_tokens", "premise_lemas", "premise_lemas_sem_sw"]:
    print(f"  {col:22s}: {ex[col]}")