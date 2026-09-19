import re

resp = re.match('mineração', 'mineração de dados')
print(resp)

word_regex = "\w+"
respi = re.match(word_regex, "semana de aula")
print (respi)

word_regex = "\s+"
respid = re.split(word_regex, "semana de aula")
print (respid)

word_regex = r"\!"
respir = re.split(word_regex, "semana quente! de aula")
print (respir)

word_regex = r"[a-z]\w+"
respie = re.findall(word_regex, "4 Semana Quente! De Aula")
print (respie)

