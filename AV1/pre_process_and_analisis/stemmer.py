from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

def worker_stem_chunk(args: tuple) -> list:
    """Worker para Tokenização, Remoção de Stopwords e Stemming."""
    chunk, language = args
    pt_stopwords = set(stopwords.words(language))
    negation_words = {'não', 'nao', 'nem', 'nunca', 'jamais', 'sem', 'pouco'}
    stop_words = pt_stopwords - negation_words
    stemmer = RSLPStemmer()
    
    processed_list = []
    for text in chunk:
        tokens = word_tokenize(text, language=language)
        processed_tokens = [
            stemmer.stem(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        processed_list.append(" ".join(processed_tokens))
    return processed_list