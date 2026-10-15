"""
Etapa 2 (Questão 3) - Compara os resultados da validação, escolhe a melhor configuração
de cada modelo e salva os modelos treinados (para a Questão 4 avaliar no TESTE uma única vez).

Lê:    resultados/validacao_todos.csv (etapa 1)
Grava: resultados/tabela_*.csv, resultados/melhor_config.json, modelos/<modelo>.joblib
"""
import json

import joblib
import pandas as pd

from q3_utils import (PASTA_MODELOS, PASTA_RESULTADOS, MODELOS, ModeloTreinado, Representacao,
                      ajustar_modelo, carregar_dados, criar_modelo, exigir_arquivo, metricas,
                      parse_ngrama)

PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
res = pd.read_csv(exigir_arquivo(PASTA_RESULTADOS / "validacao_todos.csv", "Rode antes a etapa 1."))
base = res[res["modelo"] == "Baseline"].iloc[0]
res = res[res["modelo"] != "Baseline"].copy()
COLS = ["acuracia", "precisao_macro", "recall_macro", "f1_macro",
        "f1_NONE", "f1_ENTAILMENT", "f1_PARAPHRASE"]


def melhor_por(grupo):
    """Linha de maior F1 macro dentro de cada grupo (empates: a primeira)."""
    return res.loc[res.groupby(grupo)["f1_macro"].idxmax()]


def mostrar(titulo, tabela, arquivo):
    tabela = tabela.round(3)
    tabela.to_csv(PASTA_RESULTADOS / arquivo, index=False)
    print(f"\n=== {titulo} ===")
    print(tabela.to_string(index=False))
    print(f"  [tabela] {arquivo}")


# 1) Melhor configuração de cada modelo
melhores = melhor_por("modelo").set_index("modelo").loc[MODELOS].reset_index()
mostrar("Melhor configuração de cada modelo (validação)",
        melhores[["modelo", "vetorizador", "ngrama", "atributos_relacao", "estrategia", "params"] + COLS],
        "tabela_melhores_por_modelo.csv")
print(f"(referência: sempre NONE -> acurácia {base['acuracia']:.3f}, F1 macro {base['f1_macro']:.3f})")

# 2) Estratégias de desbalanceamento (melhor F1 macro de cada modelo x estratégia)
est = melhor_por(["modelo", "estrategia"])
est = est.set_index("modelo").loc[MODELOS].reset_index()
mostrar("Estratégias de desbalanceamento (melhor de cada modelo x estratégia)",
        est[["modelo", "estrategia", "vetorizador", "ngrama", "atributos_relacao"] + COLS],
        "tabela_estrategias.csv")

# 3) TF-IDF x CountVectorizer
melhor_vec = melhor_por(["modelo", "vetorizador"])
melhor_vec = melhor_vec.set_index("modelo").loc[MODELOS].reset_index()
mostrar("TF-IDF x CountVectorizer (melhor de cada um, por modelo)",
        melhor_vec[["modelo", "vetorizador", "ngrama", "atributos_relacao", "estrategia", "params"] + COLS],
        "tabela_tfidf_vs_count.csv")

chave = ["modelo", "ngrama", "atributos_relacao", "estrategia", "params"]
a = res[res["vetorizador"] == "tfidf"].set_index(chave)["f1_macro"]
b = res[res["vetorizador"] == "count"].set_index(chave)["f1_macro"]
par = pd.concat([a.rename("tfidf"), b.rename("count")], axis=1).dropna().reset_index()
par["dif"] = par["tfidf"] - par["count"]
pareado = par.groupby("modelo").agg(configs_pareadas=("dif", "size"), f1_tfidf_medio=("tfidf", "mean"),
                                    f1_count_medio=("count", "mean"), dif_media=("dif", "mean"),
                                    tfidf_ganha=("dif", lambda d: int((d > 0).sum()))).loc[MODELOS].reset_index()
mostrar("Comparação pareada: mesmas configurações, só muda o vetorizador (F1 macro)",
        pareado, "tabela_tfidf_vs_count_pareado.csv")

# 4) Efeito dos atributos de relação
rel = melhor_por(["modelo", "atributos_relacao"])
rel = rel.set_index("modelo").loc[MODELOS].reset_index()
mostrar("Sem x com atributos de relação (melhor de cada)",
        rel[["modelo", "atributos_relacao", "vetorizador", "ngrama", "estrategia"] + COLS],
        "tabela_atributos_relacao.csv")

# 5) Reajusta a melhor configuração de cada modelo no treino e salva
treino, validacao, over, pesos = carregar_dados()
y_va = validacao["entailment_judgment"].to_numpy()
config_final = {}
print("\n=== Reajustando e salvando os melhores modelos ===")
for _, r in melhores.iterrows():
    cfg = dict(modelo=r["modelo"], vetorizador=r["vetorizador"], ngrama=r["ngrama"],
               atributos_relacao=bool(r["atributos_relacao"]), estrategia=r["estrategia"],
               params=json.loads(r["params"]))
    rep = Representacao(cfg["vetorizador"], parse_ngrama(cfg["ngrama"]), cfg["atributos_relacao"]).ajustar(treino)
    dados_ajuste = over if cfg["estrategia"] == "oversampling" else treino
    modelo = criar_modelo(cfg["modelo"], cfg["params"], cfg["estrategia"], pesos)
    ajustar_modelo(modelo, cfg["modelo"], rep.transformar(dados_ajuste),
                   dados_ajuste["entailment_judgment"].to_numpy(), cfg["estrategia"], pesos)
    pacote = ModeloTreinado(cfg["modelo"], cfg, rep, modelo)
    f1 = metricas(y_va, pacote.prever(validacao))["f1_macro"]
    joblib.dump(pacote, PASTA_MODELOS / f"{cfg['modelo']}.joblib")
    cfg["f1_macro_validacao"] = f1
    config_final[cfg["modelo"]] = cfg
    print(f"  {cfg['modelo']:20s} {cfg['vetorizador']:5s} ngrama={cfg['ngrama']} "
          f"rel={str(cfg['atributos_relacao']):5s} {cfg['estrategia']:12s} {cfg['params']}  "
          f"F1 macro validação={f1:.3f}  -> modelos/{cfg['modelo']}.joblib")

melhor_geral = max(config_final.values(), key=lambda c: c["f1_macro_validacao"])["modelo"]
with open(PASTA_RESULTADOS / "melhor_config.json", "w", encoding="utf-8") as f:
    json.dump({"melhor_geral_na_validacao": melhor_geral, "configuracoes": config_final},
              f, indent=2, ensure_ascii=False)
print(f"\nMelhor modelo na validação: {melhor_geral}  (resultados/melhor_config.json)")
print("O teste NÃO foi usado. Ele será avaliado uma única vez na Questão 4.")
