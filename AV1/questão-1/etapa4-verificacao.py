"""
Etapa 4 - Verificação.
"""
import json
import re
import pandas as pd
from preprocessamento import (PASTA_SAIDA, PROTEGIDAS, SPLITS, carregar_processado,
                              carregar_split, exigir_arquivo)

falhas = 0


def checar(descricao, condicao):
    global falhas
    print(("OK    " if condicao else "FALHA ") + descricao)
    if not condicao:
        falhas += 1


NEGACOES = {"não", "nem", "nunca", "jamais", "nenhum", "nenhuma", "ninguém", "nada", "sem"}
COLS_NOVAS = ["norm", "tokens", "lemas", "lemas_sem_sw"]

for split in SPLITS:
    print(f"\n--- {split} ---")
    bruto = carregar_split(split)
    proc = carregar_processado(f"{split}_processado.parquet")

    checar("mesmo número de linhas do original", len(proc) == len(bruto))
    checar("sentence_pair_id igual e na mesma ordem",
           proc["sentence_pair_id"].tolist() == bruto["sentence_pair_id"].tolist())
    checar("rótulos entailment_judgment inalterados",
           proc["entailment_judgment"].tolist() == bruto["entailment_judgment"].tolist())
    checar("relatedness_score inalterado",
           proc["relatedness_score"].tolist() == bruto["relatedness_score"].tolist())
    checar("premise/hypothesis originais preservadas",
           proc["premise"].tolist() == bruto["premise"].tolist()
           and proc["hypothesis"].tolist() == bruto["hypothesis"].tolist())

    for col in ["premise", "hypothesis"]:
        for sufixo in COLS_NOVAS:
            nome = f"{col}_{sufixo}"
            checar(f"{nome}: sem nulos", int(proc[nome].isna().sum()) == 0)
        for sufixo in ["norm", "tokens", "lemas"]:
            checar(f"{col}_{sufixo}: nenhuma frase vazia",
                   int((proc[f"{col}_{sufixo}"].str.strip() == "").sum()) == 0)
        checar(f"{col}_lemas_sem_sw: nenhuma frase vazia",
               int((proc[f"{col}_lemas_sem_sw"].str.strip() == "").sum()) == 0)

        # negações: quantas existem em 'lemas' vs. quantas sobrevivem em 'lemas_sem_sw'
        # (o que muda de 'lemas' para 'lemas_sem_sw' é só a remoção de stopwords)
        em_tokens = sum(len(NEGACOES & set(t.split())) for t in proc[f"{col}_lemas"])
        em_sem_sw = sum(len(NEGACOES & set(t.split())) for t in proc[f"{col}_lemas_sem_sw"])
        checar(f"{col}: negações preservadas ao remover stopwords ({em_tokens} -> {em_sem_sw})",
               em_tokens == em_sem_sw)

        # a normalização promete remover estes ruídos: conferir em TODOS os splits
        for rotulo, p in {"aspas/travessões curvos": "[“”‘’–—]", "espaço duplo ou borda": r"^\s|\s$|\s{2,}",
                          "URL": r"https?://|www\.", "HTML": r"<[^>]+>", "@ ou #": r"[@#]\w+"}.items():
            checar(f"{col}_norm: sem {rotulo}", not proc[f"{col}_norm"].str.contains(p, regex=True).any())

        # todas as palavras PROTEGIDAS (não só negações) devem sobreviver à remoção de stopwords
        prot_lemas = sum(len(PROTEGIDAS & {w for w in s.split()}) for s in proc[f"{col}_lemas"])
        prot_sw = sum(len(PROTEGIDAS & {w for w in s.split()}) for s in proc[f"{col}_lemas_sem_sw"])
        checar(f"{col}: palavras PROTEGIDAS preservadas ({prot_lemas} frases-palavra -> {prot_sw})", prot_lemas == prot_sw)

        # pontuação comum não deve sobrar (símbolos como %, ° e + são mantidos de propósito)
        checar(f"{col}_tokens: sem pontuação comum solta",
               not proc[f"{col}_tokens"].apply(lambda s: any(re.fullmatch(r"[.,;:!?()\[\]\"'\-…/]+", w) for w in s.split())).any())

print("\n--- desbalanceamento ---")
pesos = json.load(open(exigir_arquivo(PASTA_SAIDA / "class_weights.json", "Rode antes a etapa3-desbalanceamento.py."), encoding="utf-8"))
checar("class_weights.json tem 3 classes", set(pesos) == {"0", "1", "2"})
checar("classe minoritária tem o maior peso", pesos["2"] > pesos["1"] > pesos["0"])
over = carregar_processado("train_oversampled.parquet")
cont = over["entailment_judgment"].value_counts()
checar("train_oversampled: 3 classes com o mesmo tamanho", cont.nunique() == 1)
orig_treino = carregar_split("train")
checar("train_oversampled: só contém pares que existem no treino original",
       set(zip(over["premise"], over["hypothesis"])) <= set(zip(orig_treino["premise"], orig_treino["hypothesis"])))

print("\n--- antes x depois (treino, para inspeção visual) ---")
proc = carregar_processado("train_processado.parquet")
for i in [1406, 2368, 642]:
    r = proc.iloc[i]
    print(f"\n[{i}] classe {r['entailment_judgment']}")
    print("  original :", r["premise"])
    print("  norm     :", r["premise_norm"])
    print("  lemas    :", r["premise_lemas"])
    print("  sem_sw   :", r["premise_lemas_sem_sw"])

print("\n--- tamanho do vocabulário (treino) ---")
for sufixo in COLS_NOVAS:
    vocab = set()
    for col in ["premise", "hypothesis"]:
        for s in proc[f"{col}_{sufixo}"]:
            vocab.update(s.split())
    palavras = sum(len(s.split()) for c in ["premise", "hypothesis"] for s in proc[f"{c}_{sufixo}"])
    print(f"{sufixo:13s} vocabulário={len(vocab):6d}  palavras no total={palavras}")

print("\nRESULTADO:", "tudo OK" if falhas == 0 else f"{falhas} falha(s)")