# Import Counter and word_tokenize
from collections import Counter
from nltk.tokenize import word_tokenize

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