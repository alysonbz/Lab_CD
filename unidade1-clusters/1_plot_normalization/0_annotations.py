### Aula 01 ####

# Single Linkage (Ligação Simples) = A "menor distância" entre qualquer ponto do cluster 1 e qualquer ponto do cluster 2.

# Complete Linkage (Ligação Completa) = A "maior distância" entre qualquer ponto do cluster 1 e qualquer ponto do cluster 2.

# Avarage Linkage (Ligação Média): A média de todas as distâncias entre os pontos do cluster 1 e do cluster.

#### Aula 02 - Kmeans ###

### Unidade 02 NLP
# tentar extrair características textuais
# NLP permite que as tarefas mais simples não fiquem escravas de ferramentas mais robustas (GPT)
# Pode-se identificar padrões textuais, pré-processamneto, classificação de textos, identificação de erro, análise de sentimentos
# vetorização de palavras - transformar palavras em identificadores numéricos
#
#
#
# A biblioteca nativa do python (re -> Expressões Regulares) permite buscar, validae ou manipular padrões dentro de strings.

import re
resp = re.match('mineracao', 'mineracao de dados')
print(resp)

# pode informar padrões (ex: palavra)
word_regex = "\w+"
resp = re.match(word_regex, "semana de aula")
print(resp) # Irá retornar a primeira palavra que encontrou anterior ao espaço.

word_regex = "\w+"
resp = re.match(word_regex, "semana de aula")
print(resp) # Irá retornar a primeira palavra que encontrou anterior ao espaço.

word_regex = "\w"
resp = re.match(word_regex, "s emana de aula")
print(resp) # Irá retornar a primeira palavra que encontrou anterior ao espaço.

word_regex = "\d"
resp = re.match(word_regex, "42semana de aula")
print(resp) # Irá retornar/encontrar o primeiro dígito numérico.

word_regex = "\d+"
resp = re.match(word_regex, "42semana de aula")
print(resp) # Irá retornar/encontrar um ou mais dígitos numéricos.

word_regex = "\s+" # Encontra um ou mais espaços em branco.
resp = re.split(word_regex, "42semana de aula")
print(resp) # Irá fazer a separação da string analisada. Dessa forma, ele irá retornar as palavras onde o padrão (espaços neste exemplo) forem encontrados.

word_regex = r"\!"
resp = re.split(word_regex, "42!semana de aula")
print(resp)

# match verifica se o padrão indicado ocorre exatamente no início da string.
# findall varre uma string e retorna todas as correspondências que encontrar.
# split faz uma divisão de string.
# search faz uma busca. (Percorre a string inteira procurando a primeira ocorrência do padrão.)

word_regex = r"[a-z]\w+"
resp = re.findall(word_regex, "4 Semana Quente! De Aula")
print(resp) # Irá percorrer a string inteira e irá retornar uma lista de strings com letras minúsculas

word_regex = r"[a-z]\w+" # Faz uma busca de "a" até "z".
resp = re.match(word_regex, "4 Semana Quente! De Aula")
print(resp) # Não retornou nada porque começa com número.

word_regex = r"[a-z]\w+"
resp = re.search(word_regex, "4 Semana Quente! De Aula")
print(resp)
