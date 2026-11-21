# Import Dictionary
from gensim.corpora.dictionary import Dictionary

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils import get_pre_process_wiki_articles

# Create a Dictionary from the articles: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
word = ""
for item in dictionary.token2id.items():
    if item[1] == computer_id:
        word = item[0]
        break
print('the word', word, 'has index', computer_id, 'in dictionary')

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[0][:10])
