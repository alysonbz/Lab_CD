import re

resp = re.match('mineracao', 'mineracao de dados')
print(resp)

word_regex = "\w+"
resp = re.match(word_regex, "semana de aula")
print(resp)

word_regex = "\d"
resp = re.match(word_regex, "digito de aula")
print(resp)

word_regex = r"\!"
resp = re.split(word_regex, "semana quente! de aula")
print(resp)

word_regex = r"[a-z]\w+"
resp = re.match(word_regex, " 4 Semanas Quente! De Aula")
print(resp)

word_regex = r"[a-z]\w+"
resp = re.search(word_regex, " 4 Semanas Quente! De Aula")
print(resp)