#import re
match_digits_and_words = ('(\d+|w+)')
print(re.findall(match_digits_and_words,'He has 11 cats. Do you like cats?'))

import re
match_digits_and_words = ('(\d+| cats)')
print(re.findall(match_digits_and_words,'He has 11 cats. do you like cats?'))

import re
parttern1 = '(\w+|\?| !)'
parttern2 = '(\w+|#\d+|\?|!)'
parttern3 = '(#\d\W+\?!)'
parttern4 = '\s+'
partter = '?'
print(re.findall(parttern4, "SOLIDER #1: Found them? In Mercea? The coconut's tropical!"))



from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')


text = """The cat is in the box. The cat likes the box. The box is over the cat."""
tokens = [w for  w in word_tokenize(text.lower())
            if w.isalpha()]
no_stops = [t for t in tokens
            if t not in stopwords.words('english')]
Counter().most_common(2)
print(Counter(no_stops).most_common(2))