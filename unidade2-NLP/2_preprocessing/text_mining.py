
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk.tokenize import sent_tokenize
# Importação do corpus de stop words
from nltk.corpus import stopwords

from src.utils import get_sample_article

# Garante o download das stop words do NLTK (executa apenas uma vez)
nltk.download('stopwords', quiet=True)
# Define o conjunto de stop words em inglês
stop_words = set(stopwords.words('english'))

# Carrega o artigo
my_documents = get_sample_article()

if isinstance(my_documents, list):
    article_text = " ".join(my_documents)
else:
    article_text = my_documents

# Tokenização por sentença
sentences = sent_tokenize(article_text.lower())

# Quebra por espaço E filtra as stop words de cada sentença
tokenize_docs = [
    [word for word in sent.split() if word not in stop_words]
    for sent in sentences
]

# Cria o dicionário Gensim
dictionary = Dictionary(tokenize_docs)

# Cria o corpus Bag-of-Words (BoW)
corpus = [dictionary.doc2bow(doc) for doc in tokenize_docs]

# Cria o modelo e a matriz TF-IDF
tfidf = TfidfModel(corpus)
tfidf_matrix = [tfidf[sent_bow] for sent_bow in corpus]

# Exibe o resultado da primeira sentença filtrada
print("Primeira sentença limpa (termos mapeados):", tokenize_docs[0])
print("\nPesos TF-IDF da primeira sentença:", tfidf_matrix[0])
