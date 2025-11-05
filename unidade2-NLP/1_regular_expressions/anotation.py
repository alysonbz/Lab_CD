import re

#resp = re.match('dados','mineracao de dados')

#word_regex = "\d+"
#resp = re.match(word_regex, "42semana de aula")
#print(resp)

#resp = re.split(word_regex, "semana de aula")
#print(resp)

#word_regex = r"[a-z]"
#resp = re.split(word_regex, "Semana quente! De aula")
#print(resp)

#word_regex = r"[a-z]\w+"
#resp = re.split(word_regex, "4 Semanas Quente! De Aula")
#print(resp)

#ord_regex = r"[a-z]\w+"
#resp = re.findall(word_regex, "4 Semanas Quente! De Aula")
#print(resp)

word_regex = r"[a-z]\w+"
resp = re.search(word_regex, "4 Semanas Quente! De Aula")
print(resp)

