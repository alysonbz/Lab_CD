"""
Etapa 3 (Questão 3) - Verificação. Cada check imprime OK ou FALHA.
Rode depois das etapas 1 e 2.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd

from q3_utils import (ATRIBUTOS_RELACAO_POR_MODELO, ESTRATEGIAS_POR_MODELO, MODELOS, NOMES_REL,
                      PASTA_MODELOS, PASTA_QUESTAO, PASTA_RESULTADOS, atributos_relacao,
                      carregar_dados, exigir_arquivo, metricas)

falhas = 0


def checar(descricao, condicao):
    global falhas
    print(("OK    " if condicao else "FALHA ") + descricao)
    if not condicao:
        falhas += 1


treino, validacao, over, pesos = carregar_dados()
y_tr = treino["entailment_judgment"].to_numpy()
y_va = validacao["entailment_judgment"].to_numpy()

print("--- o teste não foi usado ---")
for arq in ["q3_utils.py", "etapa1-treino-validacao.py", "etapa2-selecao-e-modelos.py"]:
    codigo = (PASTA_QUESTAO / arq).read_text(encoding="utf-8")
    checar(f"{arq}: não referencia arquivos de teste",
           not re.search(r"test_processado|test-0000|[\"']test[\"']", codigo))

print("\n--- dados de entrada ---")
pares_tr = set(zip(treino["premise"], treino["hypothesis"]))
pares_va = set(zip(validacao["premise"], validacao["hypothesis"]))
checar("nenhum par da validação aparece no treino", not (pares_tr & pares_va))
checar("oversampling só contém pares do treino original",
       set(zip(over["premise"], over["hypothesis"])) <= pares_tr)
checar("oversampling: 3 classes com o mesmo tamanho", over["entailment_judgment"].value_counts().nunique() == 1)
checar("pesos de classe = n_total / (3 x n_classe)",
       all(abs(pesos[c] - len(treino) / (3 * (y_tr == c).sum())) < 1e-9 for c in (0, 1, 2)))

print("\n--- atributos de relação (teste unitário) ---")
mini = pd.DataFrame({
    "premise_lemas_sem_sw": ["a b c 8", "x y"], "hypothesis_lemas_sem_sw": ["b c d 9", "x y"],
    "premise_lemas": ["a b c 8", "x y não"], "hypothesis_lemas": ["b c d 9", "x y"],
    "premise_tokens": ["a b c 8", "x y não"], "hypothesis_tokens": ["b c d 9", "x y"]})
f = atributos_relacao(mini)
checar("jaccard de 'a b c 8' x 'b c d 9' = 2/6", abs(f.loc[0, "jaccard"] - 2 / 6) < 1e-9)
checar("números diferentes: numeros_iguais=0 e numeros_novos_h=1",
       f.loc[0, "numeros_iguais"] == 0 and f.loc[0, "numeros_novos_h"] == 1)
checar("negação só na premissa: neg_assimetrica=1", f.loc[1, "neg_assimetrica"] == 1)
checar("frases iguais: jaccard=1", abs(f.loc[1, "jaccard"] - 1) < 1e-9)
real = atributos_relacao(validacao)
checar("atributos de relação reais sem NaN/inf", bool(np.isfinite(real.values).all()))
checar("nenhum atributo vem do rótulo, de relatedness_score ou de id",
       not any(re.search(r"relatedness|entailment|judgment|pair_id", n) for n in NOMES_REL))

print("\n--- tabela de resultados ---")
res = pd.read_csv(exigir_arquivo(PASTA_RESULTADOS / "validacao_todos.csv", "Rode antes a etapa 1."))
base = res[res["modelo"] == "Baseline"].iloc[0]
res = res[res["modelo"] != "Baseline"]
metr = ["acuracia", "precisao_macro", "recall_macro", "f1_macro", "f1_NONE", "f1_ENTAILMENT", "f1_PARAPHRASE"]
checar("métricas sem nulos e entre 0 e 1", bool(res[metr].notna().all().all()
                                                 and res[metr].min().min() >= 0 and res[metr].max().max() <= 1))
checar("os 4 modelos aparecem", set(res["modelo"]) == set(MODELOS))
checar("os 2 vetorizadores aparecem", set(res["vetorizador"]) == {"tfidf", "count"})
checar("as 3 estratégias de desbalanceamento aparecem", set(res["estrategia"]) == {"nenhuma", "pesos", "oversampling"})
checar("Naive Bayes nunca recebeu atributos de relação (valores negativos)",
       not res[res["modelo"] == "NaiveBayes"]["atributos_relacao"].any())
checar("MLP nunca recebeu 'pesos' (o scikit-learn não permite)",
       "pesos" not in set(res[res["modelo"] == "MLP"]["estrategia"]))
checar("cada modelo só usa as estratégias previstas",
       all(set(res[res["modelo"] == m]["estrategia"]) <= set(ESTRATEGIAS_POR_MODELO[m]) for m in MODELOS))
checar("baseline 'sempre NONE' tem a acurácia da proporção de NONE na validação",
       abs(base["acuracia"] - (y_va == 0).mean()) < 1e-9)
checar(f"melhor F1 macro ({res['f1_macro'].max():.3f}) supera o baseline ({base['f1_macro']:.3f})",
       res["f1_macro"].max() > base["f1_macro"])
print(f"      (informativo) configurações sem convergência: {int((~res['convergiu']).sum())} de {len(res)}")

print("\n--- modelos salvos ---")
cfg = json.load(open(exigir_arquivo(PASTA_RESULTADOS / "melhor_config.json", "Rode antes a etapa 2."), encoding="utf-8"))
checar("melhor_config.json tem os 4 modelos", set(cfg["configuracoes"]) == set(MODELOS))
# o vetorizador quebra o texto com o padrão \b\w+\b (ex.: "recém-nascido" vira "recém" e "nascido"),
# então o vocabulário de referência do treino precisa ser quebrado do mesmo jeito
vocab_tr = {w for s in treino["premise_lemas_sem_sw"].tolist() + treino["hypothesis_lemas_sem_sw"].tolist()
            for w in re.findall(r"(?u)\b\w+\b", s)}
for nome in MODELOS:
    pacote = joblib.load(exigir_arquivo(PASTA_MODELOS / f"{nome}.joblib", "Rode antes a etapa 2."))
    termos = pacote.representacao.vec.get_feature_names_out()
    fora = sorted({w for t in termos for w in t.split() if w not in vocab_tr})
    checar(f"{nome}: vocabulário só tem palavras do treino (sem vazamento)", not fora)
    if fora:
        print("      termos fora do treino (primeiros 10):", fora[:10])
    m = metricas(y_va, pacote.prever(validacao))["f1_macro"]
    esperado = cfg["configuracoes"][nome]["f1_macro_validacao"]
    checar(f"{nome}: F1 macro recalculado na validação = salvo ({m:.3f})", abs(m - esperado) < 1e-9)
    linha = res[(res["modelo"] == nome) & (res["vetorizador"] == pacote.config["vetorizador"])
                & (res["ngrama"] == pacote.config["ngrama"])
                & (res["atributos_relacao"] == pacote.config["atributos_relacao"])
                & (res["estrategia"] == pacote.config["estrategia"])
                & (res["params"] == json.dumps(pacote.config["params"]))]
    checar(f"{nome}: reproduz o valor da tabela da etapa 1",
           len(linha) == 1 and abs(linha.iloc[0]["f1_macro"] - m) < 1e-6)
    checar(f"{nome}: é o melhor F1 macro do modelo na tabela",
           abs(res[res["modelo"] == nome]["f1_macro"].max() - m) < 1e-9)
    checar(f"{nome}: tipo de atributos coerente com a configuração",
           pacote.representacao.usa_rel == pacote.config["atributos_relacao"])

print("\n--- tabelas de resumo ---")
for arq in ["tabela_melhores_por_modelo.csv", "tabela_estrategias.csv", "tabela_tfidf_vs_count.csv",
            "tabela_tfidf_vs_count_pareado.csv", "tabela_atributos_relacao.csv"]:
    checar(f"{arq} existe e não está vazia", (PASTA_RESULTADOS / arq).is_file()
           and len(pd.read_csv(PASTA_RESULTADOS / arq)) > 0)

print("\nRESULTADO:", "tudo OK" if falhas == 0 else f"{falhas} falha(s)")