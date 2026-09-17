import re
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
partter = ?
print(re.findall(parttern4, "SOLIDER #1: Found them? In Mercea? The coconut's tropical!"))