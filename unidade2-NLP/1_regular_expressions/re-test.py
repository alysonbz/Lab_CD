import re

match_digits_and_words = ('\\d+|\\w+')
print(re.findall(match_digits_and_words, 'He has 11 cats. Do you like cats?'))


match_digits_and_words = ('\\d+|cats')
print(re.findall(match_digits_and_words, 'He has 11 cats. Do you like cats?'))

pattern1 = '(\\w+|\\?|!)'
pattern2 = '(\\w+|#\\d+|\\?|!)'
pattern3 = "(#\\d|\\w+|\\?!)"
pattern4 = '\\s+'

pattern = ''

print(re.findall(pattern2, "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"))