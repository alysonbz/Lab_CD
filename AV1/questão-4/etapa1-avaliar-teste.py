"""
Etapa 1 (Questão 4) - Avaliação no TESTE (feita UMA ÚNICA VEZ).

Para cada um dos 4 modelos, avalia no teste duas configurações, ambas escolhidas na
VALIDAÇÃO (Questão 3): a melhor com TF-IDF e a melhor com CountVectorizer. A melhor das duas
é o "modelo final" do modelo; a outra serve só para comparar TF-IDF x CountVectorizer.
Nada é reajustado ou escolhido olhando o teste.

Grava: resultados/predicoes_teste.csv e resultados/configuracoes_avaliadas.csv
Para repetir de propósito (não recomendado): python etapa1-avaliar-teste.py --refazer
"""
import json
import sys

import pandas as pd

from q4_utils import PASTA_RESULTADOS, carregar_teste, q3

destino = PASTA_RESULTADOS / "predicoes_teste.csv"
if destino.exists() and "--refazer" not in sys.argv:
    raise SystemExit("O teste já foi avaliado (resultados/predicoes_teste.csv existe).\n"
                     "Avaliar o teste várias vezes e escolher o melhor resultado o torna otimista.\n"
                     "Use as etapas 2 e 3 para gerar tabelas e figuras a partir das predições salvas.")
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

treino, validacao, over, pesos = q3.carregar_dados()
teste = carregar_teste()
y_te = teste["entailment_judgment"].to_numpy()
y_va = validacao["entailment_judgment"].to_numpy()
print(f"treino={len(treino)}  validação={len(validacao)}  teste={len(teste)}\n")

candidatas = pd.read_csv(q3.exigir_arquivo(q3.PASTA_RESULTADOS / "tabela_tfidf_vs_count.csv",
                                           "Rode antes as etapas 1 e 2 da Questão 3."))
with open(q3.exigir_arquivo(q3.PASTA_RESULTADOS / "melhor_config.json"), encoding="utf-8") as f:
    escolhidas = json.load(f)["configuracoes"]

previsoes = pd.DataFrame({"sentence_pair_id": teste["sentence_pair_id"].to_numpy(), "y_real": y_te})
configs = []
for _, r in candidatas.iterrows():
    nome, vet, estr = r["modelo"], r["vetorizador"], r["estrategia"]
    rel = str(r["atributos_relacao"]) == "True"
    params = json.loads(r["params"])
    rep = q3.Representacao(vet, q3.parse_ngrama(r["ngrama"]), rel).ajustar(treino)  # só no treino
    dados = over if estr == "oversampling" else treino
    modelo = q3.criar_modelo(nome, params, estr, pesos)
    q3.ajustar_modelo(modelo, nome, rep.transformar(dados),
                      dados["entailment_judgment"].to_numpy(), estr, pesos)
    chave = f"{nome}|{vet}"
    previsoes[chave] = modelo.predict(rep.transformar(teste))
    f1_va = q3.metricas(y_va, modelo.predict(rep.transformar(validacao)))["f1_macro"]
    f1_te = q3.metricas(y_te, previsoes[chave].to_numpy())["f1_macro"]
    configs.append(dict(chave=chave, modelo=nome, vetorizador=vet, ngrama=r["ngrama"],
                        atributos_relacao=rel, estrategia=estr, params=r["params"],
                        escolhida_na_validacao=(escolhidas[nome]["vetorizador"] == vet),
                        f1_macro_validacao=f1_va))
    print(f"{chave:28s} ngrama={r['ngrama']} rel={str(rel):5s} {estr:12s} {r['params']:18s} "
          f"F1 macro: validação={f1_va:.3f}  teste={f1_te:.3f}")

previsoes.to_csv(destino, index=False)
pd.DataFrame(configs).to_csv(PASTA_RESULTADOS / "configuracoes_avaliadas.csv", index=False)
print("\n[salvo] resultados/predicoes_teste.csv e resultados/configuracoes_avaliadas.csv")
print("A partir daqui, use só as predições salvas (etapas 2 e 3). Não rode o teste de novo.")
