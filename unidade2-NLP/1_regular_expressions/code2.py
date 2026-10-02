# Import necessary modules
from nltk.tokenize import sent_tokenize , word_tokenize
from src.utils import get_sample_Santo_Graal

# Split scene_one into sentences: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(scene_one)

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(sentences)

# Print the unique tokens result
print(unique_tokens)