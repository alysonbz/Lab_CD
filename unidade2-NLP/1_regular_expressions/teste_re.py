import re

resp = re.match('mineracao','mineracao de dados')

print(resp)

resp = re.match('dados','mineracao de dados')

print(resp)

word_regex = "\w+"

resp = re.match(word_regex,'semana de aula')

print(resp)

word_regex = "\s+"

resp = re.split(word_regex, "semana de aula")

print(resp)

word_regex = r"[a-z}\w+"
resp = re.findall(word_regex, "4  Semanas Quentes! De Aula")
print(resp)