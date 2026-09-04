import matplotlib.pyplot as plt
import re
from nltk.tokenize import regexp_tokenize

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils import get_sample_Santo_Graal

# Split the script into lines: lines
holy_grail = get_sample_Santo_Graal()
lines = holy_grail.split(r'\n')

# Replace all script lines for speaker
pattern = r"[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(l, pattern) for l in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words, bins=range(1, 20), align='left')
plt.xticks(range(1, 20))
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.title('Distribution of Line Lengths')

# Show the plot
plt.show()