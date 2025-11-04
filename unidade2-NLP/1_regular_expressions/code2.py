# Import necessary modules
from nltk.tokenize import word_tokenize
from src.utils import get_sample_Santo_Graal
from nltk.tokenize import sent_tokenize

# Split scene_one into sentences: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])
# [len(w) for w in scene_one]
# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(sentences))

# Print the unique tokens result
print(unique_tokens)