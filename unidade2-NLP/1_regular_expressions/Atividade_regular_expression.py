import re

#1. **Contagem de Correspondências:**
#   Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões regulares.

def cont_word(text):
    regex = "Python"
    resp = re.findall(regex,text)
    return len(resp)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print("resultado primeira questão: ",cont_word(text))

#2)

def valid_email(email):
    arroba = "@"
    resp = re.findall(arroba, email)
    if len(resp)>1:
        return "email inválido"

    espaco = r"\s+"
    resp = re.findall(espaco, email)
    if len(resp) > 0:
        return "email inválido"

    particao = re.split(arroba, email)
    if len(re.split("",particao[0]))>64:
        return "email inválido"

    if ("." not in re.split("",particao[1])):
        return "email inválido"

    return "email válido"

email = 'ricardo112.41@gmail..com'
print("resultado da segunda questão: ", valid_email(email))