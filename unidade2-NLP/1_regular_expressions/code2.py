# Import necessary modules
from nltk.tokenize import word_tokenize, sent_tokenize

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    
from src.utils import get_sample_Santo_Graal

# Split scene_one into sentences: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(tokenized_sent)

# Print the unique tokens result
print(unique_tokens)