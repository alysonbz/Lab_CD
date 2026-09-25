from gensim.corpora.dictionary import Dictionary
from src.utils import get_pre_process_wiki_articles
from gensim.models import TfidfModel

# Create a Dictionary from the articles: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Get the fifth document in corpus: doc
doc = corpus[4]

# 1. Inicialize um TfidfModel chamado tfidf usando corpus como argumento
tfidf = TfidfModel(corpus)

# 2. Utilize doc para calcular os pesos: tfidf_weights
tfidf_weights = tfidf[doc]

# 3. Print os 5 primeiros elementos de tfidf_weights
print(tfidf_weights[:5])

# 4. Ordene tfidf_weights em ordem decrescente: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# 5. Utilizando dictionary, exiba os 5 maiores pesos junto com seus tokens
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)