import os
import spacy
import nltk
from nltk.corpus import stopwords
from pre_process_and_analisis.spell_checker import get_pt_dictionary_path

try:
    nlp = spacy.load("pt_core_news_sm", disable=["ner", "parser"])
except Exception:
    os.system("python -m spacy download pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm", disable=["ner", "parser"])

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Cache global por processo para o dicionário e o mapa de infinitivo O(1)
_VALID_WORDS_CACHE = None
_INF_MAPPING_CACHE = None

def _load_dictionary_caches():
    global _VALID_WORDS_CACHE, _INF_MAPPING_CACHE
    if _VALID_WORDS_CACHE is not None and _INF_MAPPING_CACHE is not None:
        return _VALID_WORDS_CACHE, _INF_MAPPING_CACHE
    
    valid_words = {
        'gostei', 'satisfeito', 'insatisfeito', 'otima', 'otimo', 'qualidade', 
        'preco', 'facil', 'bateria', 'imagem', 'imagens', 'display', 'design', 
        'wifi', 'wifis', 'gb', 'hd', 'led', 'produto', 'recomendo', 'desabone', 'rapida'
    }
    
    inf_mapping = {
        'poderia': 'poder', 'pode': 'poder', 'podia': 'poder', 'puderam': 'poder', 'pude': 'poder',
        'gostei': 'gostar', 'gostou': 'gostar', 'gostaram': 'gostar', 'gostando': 'gostar',
        'recomendo': 'recomendar', 'recomenda': 'recomendar', 'recomendamos': 'recomendar',
        'satisfeito': 'satisfeito', 'satisfeita': 'satisfeito', 'satisfeitos': 'satisfeito',
        'comprei': 'comprar', 'comprou': 'comprar', 'compramos': 'comprar', 'comprando': 'comprar',
        'funcionou': 'funcionar', 'funciona': 'funcionar', 'funcionando': 'funcionar',
        'veio': 'vir', 'vieram': 'vir', 'vindo': 'vir',
        'chegou': 'chegar', 'chegaram': 'chegar', 'chegando': 'chegar',
        'atendeu': 'atender', 'atendem': 'atender', 'atendendo': 'atender',
        'imagens': 'imagem', 'papeis': 'papel', 'papéis': 'papel'
    }
    
    try:
        dict_path = get_pt_dictionary_path()
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        word = parts[0].lower()
                        valid_words.add(word)
                        
                        # Heurística inteligente baseada no dicionário gigante: 
                        # Identifica infinitivos (-ar, -er, -ir) e mapeia variações comuns se existirem na base
                        if word.endswith(('ar', 'er', 'ir')) and len(word) > 4:
                            stem = word[:-2]
                            # Mapeia gerúndios, particípios e pretéritos comuns para o infinitivo encontrado no dicionário
                            for suffix_variant, target_ending in [('ou', 'ar'), ('ava', 'ar'), ('am', 'ar'), ('ando', 'ar'),
                                                                ('eu', 'er'), ('ia', 'er'), ('endo', 'er'),
                                                                ('iu', 'ir'), ('indo', 'ir')]:
                                if stem.endswith(target_ending[:-1]):
                                    variant = stem + suffix_variant
                                    if variant not in inf_mapping:
                                        inf_mapping[variant] = word
    except Exception:
        pass
        
    _VALID_WORDS_CACHE = valid_words
    _INF_MAPPING_CACHE = inf_mapping
    return _VALID_WORDS_CACHE, _INF_MAPPING_CACHE


def worker_lemmatize_chunk(args) -> list:
    """Worker paralelo otimizado utilizando o dicionário gigante de 300k+ palavras e mapeamento O(1) para infinitivo."""
    if isinstance(args, tuple):
        chunk = args[0]
        language = args[1] if len(args) > 1 else 'portuguese'
    else:
        chunk = args
        language = 'portuguese'

    valid_words, inf_mapping = _load_dictionary_caches()

    try:
        stop_words = set(stopwords.words(language))
    except Exception:
        stop_words = nlp.Defaults.stop_words

    negation_words = {'não', 'nao', 'nem', 'nunca', 'jamais', 'sem', 'pouco'}
    effective_stop_words = stop_words - negation_words

    lemmatized_list = []
    for text in chunk:
        if not text:
            lemmatized_list.append("")
            continue
        
        doc = nlp(text)
        lemmas = []
        
        for token in doc:
            if not token.is_alpha:
                continue
                
            token_lower = token.text.lower()
            lemma_lower = token.lemma_.lower()
            
            if lemma_lower in effective_stop_words and token_lower not in valid_words:
                continue
                
            if len(token_lower) <= 1:
                continue

            # 1. Consulta o mapa otimizado de infinitivo extraído do dicionário gigante O(1)
            if token_lower in inf_mapping:
                final_word = inf_mapping[token_lower]
            elif lemma_lower in inf_mapping:
                final_word = inf_mapping[lemma_lower]
            # 2. Valida se o lema do spaCy existe no dicionário gigante sem corrupções
            elif lemma_lower in valid_words and not lemma_lower.endswith(('eirer', 'podeiro', 'recomer')):
                final_word = lemma_lower
            # 3. Fallback para a palavra original se presente no dicionário
            elif token_lower in valid_words:
                final_word = token_lower
            else:
                final_word = lemma_lower

            lemmas.append(final_word)

        lemmatized_list.append(" ".join(lemmas))

    return lemmatized_list