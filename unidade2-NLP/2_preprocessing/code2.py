# Import WordNetLemmatizer and Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from src.utils import get_wiki_article_lower_tokens, get_english_stop_words
nltk.download('wordnet')
nltk.download('omw-1.4')

lower_tokens = get_wiki_article_lower_tokens()

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens
              if t.isalpha()]

english_stop = get_english_stop_words()
# Remove all stop words: no_stops
no_stops = [t for t in lower_tokens
            if t not in get_english_stop_words()]
pontuacao = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"``","''"]
no_punctuation = [t for t in no_stops
                  if t not in pontuacao]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_punctuation]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))
