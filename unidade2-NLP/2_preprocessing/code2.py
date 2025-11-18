# Import WordNetLemmatizer and Counter
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
from src.utils import get_wiki_article_lower_tokens, get_english_stop_words
# nltk.download('wordnet')

lower_tokens = get_wiki_article_lower_tokens()

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

english_stop = get_english_stop_words()
# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stop]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(no_stops).most_common()

# Print the 10 most common tokens
print(bow[:10])
