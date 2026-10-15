"""
Funções de pré-processamento da Questão 1 (ASSIN ptbr).
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
import spacy

PASTA_QUESTAO = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_QUESTAO.parent
PASTA_DATASET = PASTA_PROJETO / "dataset"
PASTA_SAIDA = PASTA_QUESTAO / "dados_processados"
SPLITS = ["train", "validation", "test"]
COLUNAS_TEXTO = ["premise", "hypothesis"]
MODELO_SPACY = "pt_core_news_sm"

# ----------------------------------------------------------------------
# Normalização: aspas/travessões "tipográficos" -> versões simples
# ----------------------------------------------------------------------
SUBSTITUICOES = str.maketrans({
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-",
    "\u00a0": " ",  # espaço não separável
})

# ----------------------------------------------------------------------
# Palavras que NÃO podem ser removidas como stopwords, mesmo que o spaCy
# as marque como stopwords: negação, quantificação, restrição e contraste.
# Decisão nossa; pode ser ajustada e justificada no relatório.
# ----------------------------------------------------------------------
PROTEGIDAS = {
    # negação
    "não", "nem", "nunca", "jamais", "nenhum", "nenhuma", "nenhuns",
    "nenhumas", "ninguém", "nada", "sem",
    # quantidade / intensidade
    "mais", "menos", "muito", "muita", "muitos", "muitas",
    "pouco", "pouca", "poucos", "poucas", "todo", "toda", "todos", "todas",
    "quase",
    # restrição / contraste / tempo
    "apenas", "somente", "só", "mas", "porém", "contudo", "embora",
    "ainda", "já", "antes", "depois", "até", "contra",
}


def caminho_parquet(split):
    return PASTA_DATASET / f"{split}-00000-of-00001.parquet"


def exigir_arquivo(caminho, dica=""):
    """Dá uma mensagem clara se o arquivo não existir (em vez de só 'FileNotFoundError')."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}\n{dica}".strip())
    return caminho


def carregar_split(split):
    caminho = exigir_arquivo(
        caminho_parquet(split),
        f"A pasta do dataset deveria ser: {PASTA_DATASET}",
    )
    return pd.read_parquet(caminho)


def carregar_processado(nome_arquivo):
    """Lê um Parquet de dados_processados/ (gerado pelas etapas 2 e 3)."""
    caminho = exigir_arquivo(
        PASTA_SAIDA / nome_arquivo,
        "Rode antes as etapas anteriores (etapa2-preprocessar.py e etapa3-desbalanceamento.py).",
    )
    return pd.read_parquet(caminho)


def normalizar_texto(texto):
    """Normalização leve que NÃO muda o significado e NÃO muda maiúsculas."""
    texto = unicodedata.normalize("NFC", str(texto))  # acentos em forma única
    texto = texto.translate(SUBSTITUICOES)            # aspas e travessões
    texto = re.sub(r"\s+", " ", texto).strip()        # espaços duplicados/bordas
    return texto


def carregar_nlp():
    """Carrega o modelo de português do spaCy (sem parser e sem NER: não precisamos)."""
    try:
        return spacy.load(MODELO_SPACY, exclude=["parser", "ner"])
    except OSError:
        raise SystemExit(
            f"Modelo '{MODELO_SPACY}' não encontrado. No terminal do PyCharm, rode:\n"
            f"    python -m spacy download {MODELO_SPACY}"
        )


def analisar_frases(nlp, textos):
    """
    Para cada frase devolve três strings:
      tokens          -> tokens em minúsculas, sem pontuação
      lemas           -> lemas em minúsculas, sem pontuação
      lemas_sem_sw    -> lemas sem stopwords (exceto as PROTEGIDAS)

    Obs.: o spaCy desfaz contrações no lema (ex.: "na" -> "em o", "deles" -> "de eles").
    Por isso a lista de stopwords é aplicada a cada parte do lema.
    """
    stopwords = nlp.Defaults.stop_words
    resultados = []
    for doc in nlp.pipe(textos, batch_size=256):
        tokens, lemas, lemas_sem_sw = [], [], []
        for t in doc:
            if t.is_punct or t.is_space:
                continue
            tok = t.text.lower()
            lem = (t.lemma_ or t.text).lower()
            tokens.append(tok)
            lemas.append(lem)
            if tok in PROTEGIDAS or lem in PROTEGIDAS:
                lemas_sem_sw.append(lem)
            elif not (t.is_stop or tok in stopwords):
                for parte in lem.split():
                    if parte in PROTEGIDAS or parte not in stopwords:
                        lemas_sem_sw.append(parte)
        resultados.append((" ".join(tokens), " ".join(lemas), " ".join(lemas_sem_sw)))
    return resultados


def preprocessar_dataframe(df, nlp):
    """
    Devolve uma CÓPIA do df com novas colunas para premise e hypothesis:
      <col>_norm (normalizada, minúsculas, COM pontuação), <col>_tokens, <col>_lemas, <col>_lemas_sem_sw
    As colunas originais (incluindo id, relatedness_score e label) ficam intactas.
    """
    saida = df.copy()
    for col in COLUNAS_TEXTO:
        normalizados = [normalizar_texto(t) for t in df[col].tolist()]
        saida[f"{col}_norm"] = [t.lower() for t in normalizados]
        # o spaCy recebe o texto COM maiúsculas: ajuda a achar nomes próprios e lemas
        res = analisar_frases(nlp, normalizados)
        saida[f"{col}_tokens"] = [r[0] for r in res]
        saida[f"{col}_lemas"] = [r[1] for r in res]
        saida[f"{col}_lemas_sem_sw"] = [r[2] for r in res]
    return saida