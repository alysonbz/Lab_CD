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
from weakref import finalize

resp = re.match('mineracao', 'mineracao de dados')
#print(resp)

# pode informar padrões (ex: palavra)
word_regex = "\w+"
resp = re.match(word_regex, "semana de aula")
#print(resp) # Irá retornar a primeira palavra que encontrou anterior ao espaço.

word_regex = "\w+"
resp = re.match(word_regex, "semana de aula")
#print(resp) # Irá retornar a primeira palavra que encontrou anterior ao espaço.

word_regex = "\w"
resp = re.match(word_regex, "s emana de aula")
#print(resp) # Irá retornar a primeira palavra que encontrou anterior ao espaço.

word_regex = "\d"
resp = re.match(word_regex, "42semana de aula")
#print(resp) # Irá retornar/encontrar o primeiro dígito numérico.

word_regex = "\d+"
resp = re.match(word_regex, "42semana de aula")
#print(resp) # Irá retornar/encontrar um ou mais dígitos numéricos.

word_regex = "\s+" # Encontra um ou mais espaços em branco.
resp = re.split(word_regex, "42semana de aula")
#print(resp) # Irá fazer a separação da string analisada. Dessa forma, ele irá retornar as palavras onde o padrão (espaços neste exemplo) forem encontrados.

word_regex = r"\!"
resp = re.split(word_regex, "42!semana de aula")
#print(resp)

# match verifica se o padrão indicado ocorre exatamente no início da string.
# findall varre uma string e retorna todas as correspondências que encontrar.
# split faz uma divisão de string.
# search faz uma busca. (Percorre a string inteira procurando a primeira ocorrência do padrão.)

word_regex = r"[a-z]\w+"
resp = re.findall(word_regex, "4 Semana Quente! De Aula")
#print(resp) # Irá percorrer a string inteira e irá retornar uma lista de strings com letras minúsculas

word_regex = r"[a-z]\w+" # Faz uma busca de "a" até "z".
resp = re.match(word_regex, "4 Semana Quente! De Aula")
#print(resp) # Não retornou nada porque começa com número.

word_regex = r"[a-z]\w+"
resp = re.search(word_regex, "4 Semana Quente! De Aula")
#print(resp)


# ==== Aula 03 =====
# tokenização subdividir a strings em pedações úteis e personalizados.
# Ex: Tokenização por ponto ou palavras
# mltk.tokenize

# sent_tokenize: Divisão de sentenças.
# regexp_tokenize: Tokenização com base em expressões regulares.
# TweetTokenizer: Tokenização específica para tweets, pois é possível considerar de formas diferenciadas @ e #.

# _IMPORTANTE_
# A lógica OR é representada usando o caractere | ;
# Um grupo pode ser representado entre () ;
# Um range de caracteres pode ser representado entre [] .

# Exemplos: [A-Za-z]+ = 'ABCDghijk'
#           [0-9] = 0, 1, ..., 9
#

from nltk.tokenize import word_tokenize
#print(word_tokenize("Hi there!"))

import re
match_digits_words = ('(\d+|\w+)')
#print(re.findall(match_digits_words, 'He has 11 cats. Do you like cats?'))

match_digits_words2 = ('(\d+|cats)')
#print(re.findall(match_digits_words2, 'He has 11 cats. Do you like cats?'))

pattern1 = '(\w+|\?|!)'
pattern2 = '(\w+|#\d+|\?|!)'
pattern3 = '(#\d\w+\?!)'
pattern4 = '\s+'
pattern = '?'
#print(re.findall(pattern4, "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"))

#import nltk
#nltk.download('stopwords')

# Aula 4 Bag of words (saco de palavras)
# vamos ter que tranformar em número

# stopwords - conectivos não são considerados
# usar a função da biblioteca coletions "Couter" - vai criar a frequência das palavras tokenizadas (vai criar um dicionário).
# couter.most_common() - vai retornar os termos mais frequêntes
# Boas práticas:
# boa tokenização
# palavras minúsculas
# lematização: transformar palavras com mesmo significado semântico - temos que atribuir o mesmo ID (transformar para o infinitivo) - Terão o mesmo peso em uma classificação.
# remoção de caracteres especiais
# ======== EXEMPLO ======
from collections import Counter
from nltk.corpus import stopwords
text = """the cat is in the box. the cat likes the box. 
        the box is over the cat."""
tokens = [w for w in word_tokenize(text.lower())
          if w.isalpha()]
no_stops = [t for t in tokens
            if t not in stopwords.words('english')]
print(Counter(no_stops).most_common(2)) # most_common - Exibe as palavras mais frequêntes

# ======= AULA 04 =====
# MINERAÇÃO TEXTUAL
# Buscar informações no texto que tenham valor no algoritmo
# buscar informações mais relevantes
# Exemplo clássico => frequência dos termos  tf(t, d)
# TF contagem da frequência dos termos na sentença
# IDF frequência inversa do documento
# TF-IDF = FrequÊncia do termo * frequência inversa do documento

# ATIVIDADE - PAG 84 - utilizar tudo que aprendemos. tokenizar por sentenças
#
#



# ======= AULA 05 =======
# Classificação de texto
# CountVactorizer()
