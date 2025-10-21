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

#2. **email:**
def valida_email(email):
    padrao = r'[a-z]@[a-z].[a-z]'
    resp = re.findall(padrao, email)
    if resp == None or resp == []:
        return f'{email} e invalido'
    else:
        return f'{email} e valido'

email = 'jef-te@'
print("resultado segunda questão: ",valida_email(email))

#3. **telefone:**
# def valida_telefone(telefone):
#     padrao = r'([\d+])[\d+]-[\d+]'
#     resp = re.findall(padrao, telefone)
#     if resp == None or resp == []:
#         return f'{telefone} e invalido'
#     else:
#         return f'{telefone} e valido'

# telefone = '(88) 98101-4442'
# print("resultado segunda questão: ",valida_telefone(telefone))
# def extrai_telefone(telefone):
#     padrao = r'('
#     resp = re.split(padrao, telefone)
#     return resp

# telefone = '(88) 98101-4442'
# print("resultado terceira questão: ",extrai_telefone(telefone))
