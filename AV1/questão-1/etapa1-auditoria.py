"""
Etapa 1 - Auditoria dos dados BRUTOS.
"""
import re
import pandas as pd
from preprocessamento import SPLITS, carregar_split

dfs = {s: carregar_split(s) for s in SPLITS}

print("=== 1. Tamanho, nulos, vazios, duplicatas ===")
for nome, d in dfs.items():
    vazios = ((d["premise"].str.strip() == "") | (d["hypothesis"].str.strip() == "")).sum()
    print(f"{nome:10s} linhas={len(d):5d} nulos={int(d.isna().sum().sum())} "
          f"vazios={int(vazios)} pares_duplicados={int(d.duplicated(['premise', 'hypothesis']).sum())}")

print("\n=== 2. Distribuição das classes por split (proporção) ===")
for nome, d in dfs.items():
    cont = d["entailment_judgment"].value_counts().sort_index()
    prop = (cont / len(d)).round(3).to_dict()
    print(f"{nome:10s} contagem={cont.to_dict()} proporção={prop}")

print("\n=== 3. Comprimento das frases (treino, premise+hypothesis) ===")
tr = dfs["train"]
textos = pd.concat([tr["premise"], tr["hypothesis"]]).astype(object)
print("caracteres:", textos.str.len().describe().round(1).to_dict())
print("palavras  :", textos.str.split().str.len().describe().round(1).to_dict())

print("\n=== 4. Ruído por split (nº de frases afetadas; premise+hypothesis) ===")
padroes = {
    "URL": r"https?://|www\.",
    "HTML": r"<[^>]+>",
    "@ ou #": r"[@#]\w+",
    "aspas/apóstrofos curvos": "[“”‘’]",
    "travessão/meia-risca": "[–—]",
    "espaço duplo": r"\s{2,}",
    "dígitos": r"\d",
    "R$": r"R\$",
    "%": "%",
    "parênteses": r"[()]",
    "caractere quebrado (mojibake)": "Ã|Â|�",
}
print(f"{'padrão':32s}" + "".join(f"{s:>12s}" for s in SPLITS))
for rotulo, p in padroes.items():
    linha = f"{rotulo:32s}"
    for s in SPLITS:
        t = pd.concat([dfs[s]["premise"], dfs[s]["hypothesis"]]).astype(object)
        linha += f"{int(t.apply(lambda x: bool(re.search(p, x))).sum()):12d}"
    print(linha)

print("\n=== 5. Vazamento entre splits (pares idênticos ao treino) ===")
pares_treino = set(zip(tr["premise"], tr["hypothesis"]))
for nome in ["validation", "test"]:
    d = dfs[nome]
    print(f"{nome:10s} pares em comum com o treino: {len(set(zip(d['premise'], d['hypothesis'])) & pares_treino)}")

print("\n=== 6. Negação por classe (treino): % de frases com palavra de negação ===")
neg = r"\b(não|nunca|nenhum|nenhuma|ninguém|nada|jamais|sem|nem)\b"
for c in sorted(tr["entailment_judgment"].unique()):
    d = tr[tr["entailment_judgment"] == c]
    f = lambda s: s.str.lower().astype(object).apply(lambda x: bool(re.search(neg, x))).mean()
    print(f"classe {c}: premissa={f(d['premise']):.3f} hipótese={f(d['hypothesis']):.3f}")

print("\n=== 7. relatedness_score por classe (treino) ===")
print(tr.groupby("entailment_judgment")["relatedness_score"].agg(["mean", "min", "max"]).round(2))