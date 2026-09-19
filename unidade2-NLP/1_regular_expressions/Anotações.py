import  re

match_digits_and_words = ('(\d+|cats)')
print(re.findall(match_digits_and_words, 'He has 11 cats. Do you like cats?'))

pattern1 = '(\w+|\?\!)'
