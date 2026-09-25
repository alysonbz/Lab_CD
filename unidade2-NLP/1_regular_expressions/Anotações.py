import  re

match_digits_and_words = ('(\d+|cats)')
print(re.findall(match_digits_and_words, 'He has 11 cats. Do you like cats?'))

pattern1 = '(\w+|\?\!)'

#----------------------------------------------
#       Aula dia 24/09/2026
#----------------------------------------------

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

text = """The cat is in the box. The cat likes the box.
The box is over the cat."""
tokens = [w for w in word_tokenize(text.lower())
          if w.isalpha()]
no_stops = [t for t in tokens
            if t not in stopwords.words('english')]
print(Counter(no_stops).most_common(2))

