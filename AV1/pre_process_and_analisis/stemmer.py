from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

def worker_stem_chunk(args) -> list:
    """Worker para Tokenização, Remoção de Stopwords e Stemming."""
    if isinstance(args, tuple):
        chunk = args[0]
        language = args[1] if len(args) > 1 else 'portuguese'
    else:
        chunk = args
        language = 'portuguese'

    pt_stopwords = set(stopwords.words(language))
    negation_words = {'não', 'nao', 'nem', 'nunca', 'jamais', 'sem', 'pouco'}
    stop_words = pt_stopwords - negation_words
    stemmer = RSLPStemmer()
    
    processed_list = []
    for text in chunk:
        if not text:
            processed_list.append("")
            continue
            
        tokens = word_tokenize(text, language=language)
        processed_tokens = [
            stemmer.stem(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        processed_list.append(" ".join(processed_tokens))
        
    return processed_list