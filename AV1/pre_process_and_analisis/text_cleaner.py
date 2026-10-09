import re
import unicodedata
import nltk

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('portuguese'))
except Exception:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('portuguese'))

def _normalize_word(w):
    nfkd = unicodedata.normalize('NFD', w)
    return "".join(c for c in nfkd if unicodedata.category(c) != 'Mn')

NORMALIZED_STOP_WORDS = {_normalize_word(sw) for sw in STOP_WORDS}

NEGATION_WORDS = {'nao', 'nunca', 'jamais', 'nem', 'sem'}
EFFECTIVE_STOP_WORDS = NORMALIZED_STOP_WORDS - NEGATION_WORDS


def worker_clean_chunk(chunk: list) -> list:
    """Worker para limpeza básica, normalização, expansão de rótulos e remoção de stop words."""
    cleaned_list = []
    for text in chunk:
        if not isinstance(text, str):
            cleaned_list.append("")
            continue
            
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = "".join(c for c in text if unicodedata.category(c) != 'Mn')
        
        text = re.sub(r'o que gostei:', 'gostei', text)
        text = re.sub(r'o que nao gostei:', 'nao_gostei', text)
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        filtered_words = [word for word in words if word not in EFFECTIVE_STOP_WORDS and len(word) > 1]
        
        cleaned_list.append(" ".join(filtered_words))
        
    return cleaned_list