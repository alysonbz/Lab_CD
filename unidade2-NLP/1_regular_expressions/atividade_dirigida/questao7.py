import re

def extract_words(text):
    return re.findall(r"\b\w+\b", text)

text = "Olá, mundo! Este é um teste de extração de palavras."
print(extract_words(text))