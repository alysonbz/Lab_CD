# Import Dictionary
from gensim.corpora.dictionary import Dictionary
from src.utils import get_pre_process_wiki_articles

# Create a Dictionary from the articles: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get('computer')

# Use computer_id with the dictionary to print the word
print('the word', dictionary.get(computer_id), 'has index', computer_id, 'in dictionary')

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])
