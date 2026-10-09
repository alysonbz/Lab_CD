import spacy

try:
    nlp = spacy.load("pt_core_news_sm", disable=["ner", "parser"])
except Exception:
    import os
    os.system("python -m spacy download pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm", disable=["ner", "parser"])


def worker_lemmatize_chunk(args) -> list:
    """Worker paralelo para Tokenização, Remoção de Stopwords e Lematização via spaCy."""
    if isinstance(args, tuple):
        chunk = args[0]
        language = args[1] if len(args) > 1 else 'portuguese'
    else:
        chunk = args
        language = 'portuguese'

    stop_words = nlp.Defaults.stop_words
    negation_words = {'não', 'nao', 'nem', 'nunca', 'jamais', 'sem', 'pouco'}
    effective_stop_words = stop_words - negation_words

    lemmatized_list = []
    for text in chunk:
        if not text:
            lemmatized_list.append("")
            continue
        
        doc = nlp(text)
        lemmas = [
            token.lemma_.lower() 
            for token in doc 
            if token.is_alpha and token.lemma_.lower() not in effective_stop_words and len(token.lemma_) > 1
        ]
        lemmatized_list.append(" ".join(lemmas))

    return lemmatized_list