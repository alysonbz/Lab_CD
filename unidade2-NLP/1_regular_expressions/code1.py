import re

# Write a pattern to match sentence endings: sentence_endings
my_string = "Let's write RegEx!  Won't that be fun?  " \
            "I sure think so.  Can you find 4 sentences? " \
            "Or perhaps, all 19 words?"

# Split my_string on sentence endings and print the result
print()
print("Questão 1:")
sentence_endings = r"[.?!]"
print(re.split(sentence_endings, my_string))
print()

# Find all capitalized words in my_string and print the result
print("Questão 2:")
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))
print()

# Split my_string on spaces and print the result
print("Questão 3:")
spaces = r"\s+"
print(re.split(spaces, my_string))
print()

# Find all digits in my_string and print the result
print("Questão 4:")
digits = r"\d+"
print(re.findall(digits, my_string))