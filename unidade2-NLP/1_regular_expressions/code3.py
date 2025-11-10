
import re
from nltk.tokenize import sent_tokenize, TweetTokenizer
from src.utils import get_sample_Santo_Graal, get_tweets_sample

# === 1. Texto do Santo Graal ===
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search(r"coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())
print(match)

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*?\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"^\[.*?\]"
print(re.match(pattern2, sentences[3]))