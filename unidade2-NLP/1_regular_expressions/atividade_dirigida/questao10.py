import re

def count_vowels(text):
    vowels = re.findall(r"[aeiouáéíóúãõâêîôûàèìòù]", text, re.IGNORECASE)
    return len(vowels)

text = "Olá, mundo! Programação em Python é excelente."
print(count_vowels(text))