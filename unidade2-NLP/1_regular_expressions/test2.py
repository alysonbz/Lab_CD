import re
text = "He has 11 cats, do you like cats/?"

match_digits_and_words = ('(\d+|\w+)')
print(re.findall(match_digits_and_words, text))


match_digits_and_words = ('(\d+|cats)')
print(re.findall(match_digits_and_words, text))

pattern2 = '(\w+|#\d+|\?|!)'
print(re.findall(pattern2, "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"))