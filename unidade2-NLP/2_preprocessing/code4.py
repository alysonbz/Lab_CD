import itertools
from collections import defaultdict
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

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Get the fifth document in corpus: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    word = ""
    for item in dictionary.token2id.items():
        if item[1] == word_id:
            word = item[0]
            break
    print("The token ",word,"appears ",word_count , "times")

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Choose a key between 0 and 10 and show the count with a print function.
key = 8
print("the key", key,"in defaultdict has count: ", total_word_count[key],'\n')

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    word = ""
    for item in dictionary.token2id.items():
        if item[1] == word_id:
            word = item[0]
            break
    print(word, word_count)

