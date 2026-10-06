import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE

class TextPreprocessor:
    """
    Classe responsável pelo pré-processamento, limpeza, tokenização,
    stemming e vetorização de corpora textuais, além do tratamento de desbalanceamento.
    """
    def __init__(self, language: str = 'portuguese'):
        self.language = language
        self._ensure_nltk_downloads()
        self.stop_words = set(stopwords.words(self.language))
        self.stemmer = RSLPStemmer()
        self.vectorizer_count = None
        self.vectorizer_tfidf = None

    def _ensure_nltk_downloads(self):
        """Garante o download transparente dos pacotes necessários do NLTK."""
        resources = ['punkt', 'stopwords', 'rslp']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource == 'stopwords' else f'stemmers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def clean_text(self, text: str) -> str:
        """Remove URLs, menções, pontuações e números, normalizando para minúsculas."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^a-záéíóúâêîôûãõç\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_stem(self, text: str) -> str:
        """Tokeniza o texto, remove stopwords e reduz as palavras ao seu radical (stem)."""
        tokens = word_tokenize(text, language=self.language)
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(processed_tokens)

    def preprocess_corpus(self, series: pd.Series) -> pd.Series:
        """Aplica o pipeline completo de limpeza, tokenização e stemming."""
        print("[INFO] Executando limpeza e normalização textual...")
        cleaned = series.astype(str).apply(self.clean_text)
        print("[INFO] Executando tokenização e stemming...")
        processed = cleaned.apply(self.tokenize_and_stem)
        return processed

    def transform_to_features(self, corpus: pd.Series, method: str = 'tfidf', max_features: int = 5000):
        """Converte o corpus textual em formato numérico (Bag-of-Words ou TF-IDF)."""
        if method == 'count':
            self.vectorizer_count = CountVectorizer(max_features=max_features)
            return self.vectorizer_count.fit_transform(corpus)
        elif method == 'tfidf':
            self.vectorizer_tfidf = TfidfVectorizer(max_features=max_features)
            return self.vectorizer_tfidf.fit_transform(corpus)
        else:
            raise ValueError("O método escolhido deve ser 'count' ou 'tfidf'.")

    def handle_imbalance(self, X, y):
        """Aplica SMOTE para sintetizar dados e balancear as classes no espaço de features."""
        print("[INFO] Aplicando SMOTE para balanceamento das classes...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled