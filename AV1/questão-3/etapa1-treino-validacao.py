"""
Etapa 1 (Questão 3) - Treina todas as combinações e avalia na VALIDAÇÃO.

Combinações: 4 modelos x 2 vetorizadores (TF-IDF, CountVectorizer) x 2 n-gramas
             x sem/com atributos de relação x estratégias de desbalanceamento
             (nenhuma, pesos de classe, oversampling) x hiperparâmetros.
O TESTE não é lido aqui (fica para a Questão 4).
Grava: resultados/validacao_todos.csv
"""
import json
import time

import pandas as pd
from sklearn.dummy import DummyClassifier

from q3_utils import (ATRIBUTOS_RELACAO_POR_MODELO, ESTRATEGIAS_POR_MODELO, GRADE, MODELOS,
                      NGRAMAS, PASTA_RESULTADOS, VETORIZADORES, Representacao,
                      ajustar_modelo, carregar_dados, criar_modelo, formata_ngrama, metricas)

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
treino, validacao, over, pesos = carregar_dados()
y_tr, y_va, y_ov = (d["entailment_judgment"].to_numpy() for d in (treino, validacao, over))
print(f"treino={len(treino)}  validação={len(validacao)}  treino_oversampled={len(over)}")
print(f"pesos de classe (do class_weights.json): {pesos}\n")

linhas = []

# Linha de referência: sempre prever a classe mais comum (NONE)
base = DummyClassifier(strategy="most_frequent").fit(treino[["premise"]], y_tr)
linhas.append(dict(modelo="Baseline", vetorizador="-", ngrama="-", atributos_relacao=False,
                   estrategia="-", params="{}", n_atributos=0, convergiu=True, tempo_s=0.0,
                   **metricas(y_va, base.predict(validacao[["premise"]]))))

inicio = time.time()
for vet in VETORIZADORES:
    for ng in NGRAMAS:
        for rel in (False, True):
            rep = Representacao(vet, ng, rel).ajustar(treino)   # ajustado SÓ no treino original
            X_tr, X_va, X_ov = rep.transformar(treino), rep.transformar(validacao), rep.transformar(over)
            print(f"[{vet:5s} ngrama={formata_ngrama(ng)} relação={str(rel):5s}] "
                  f"{X_tr.shape[1]} atributos  ({time.time() - inicio:.0f}s)")
            for nome in MODELOS:
                if rel not in ATRIBUTOS_RELACAO_POR_MODELO[nome]:
                    continue
                for estr in ESTRATEGIAS_POR_MODELO[nome]:
                    X, y = (X_ov, y_ov) if estr == "oversampling" else (X_tr, y_tr)
                    for params in GRADE[nome]:
                        t0 = time.time()
                        modelo = criar_modelo(nome, params, estr, pesos)
                        nao_convergiu = ajustar_modelo(modelo, nome, X, y, estr, pesos)
                        linhas.append(dict(
                            modelo=nome, vetorizador=vet, ngrama=formata_ngrama(ng),
                            atributos_relacao=rel, estrategia=estr,
                            params=json.dumps(params), n_atributos=X_tr.shape[1],
                            convergiu=not nao_convergiu, tempo_s=round(time.time() - t0, 2),
                            **metricas(y_va, modelo.predict(X_va))))

res = pd.DataFrame(linhas)
res.to_csv(PASTA_RESULTADOS / "validacao_todos.csv", index=False)
print(f"\n{len(res) - 1} configurações treinadas em {time.time() - inicio:.0f}s "
      f"-> resultados/validacao_todos.csv")
print(f"configurações que não convergiram: {int((~res['convergiu']).sum())}")

print("\nLinha de referência (sempre NONE):")
print(res.iloc[0][["acuracia", "f1_macro"]].round(3).to_string())
print("\nTop 10 por F1 macro na validação:")
cols = ["modelo", "vetorizador", "ngrama", "atributos_relacao", "estrategia", "params",
        "acuracia", "f1_macro", "f1_PARAPHRASE"]
print(res.iloc[1:].sort_values("f1_macro", ascending=False).head(10)[cols].round(3).to_string(index=False))
