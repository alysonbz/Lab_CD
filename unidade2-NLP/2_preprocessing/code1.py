# Import Counter and word_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter

from src.utils import get_sample_article

article = get_sample_article()

# Tokenize the article: tokens
tokens = word_tokenize(article.lower())

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t for t in tokens]
# 'C:\Users\jefte\projetos em python\ufc 2025 a 2026\lab_cd\Lab_CD\src'
# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple)