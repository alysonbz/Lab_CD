# Import Counter and word_tokenize
from jsonschema.benchmarks.import_benchmark import import_time
from nltk import word_tokenize
from collections import Counter

from src.utils import get_sample_article

article = get_sample_article()

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))

# Não é possível identificar do que se trata o artigo, pois,
# os elementos mais recorrentes são conectivos e pontuações.
# Dessa forma, seria necessário adicionar o stopwords.

#import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')