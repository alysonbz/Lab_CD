import re
match_digits_and_words = ('(\d+|\w+)')
print(re.findall(match_digits_and_words, 'He has 11 cats. Do you like cats?'))

import re
match_digits_and_words = ('(\d+|cats)')
print(re.findall(match_digits_and_words, 'He has 11 cats. Do you like cats?'))

import re
pattern1 = '(\w+|\?|!)'
pattern2 = '(\w+|#\d+|\?|!)'
pattern3 = '(#\d\w+\?!)'
pattern4 = '\s+'
pattern = '?'
print(re.findall(pattern4, "SOLDIER #1: Found then? In Mercea? The coconut's tropical!"))

