import re

def count_word(text):
    regex = r"\bPython\b"
    matches = re.findall(regex, text)
    return len(matches)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print("The word 'Python' appears", count_word(text), "times in the text.")