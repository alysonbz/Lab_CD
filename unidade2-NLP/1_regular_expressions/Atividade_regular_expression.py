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

#2. Validação de Email
def validar_email(email):
    padrao = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(padrao, email))

print(validar_email("teste@gmail.com"))
print(validar_email("invalido@com"))

#3. Extração de números de telefone
texto = "Numero um: 99999-1234 numero dois: 98888-4321."
padrao = '\d{4,5}-\d{4}'

telefones = re.findall(padrao, texto)
print(telefones)

#4. Substituição de Palavras
texto = "O gato é bonito. O gato é peludo."
novo_texto = re.sub('gato', 'cachorro', texto)
print(novo_texto)

#5. Extração de urls
texto = "Acesse https://www.google.com ou http://github.com."
padrao = 'https?://[^\s]+'

urls = re.findall(padrao, texto)
print(urls)

#6. Verificação de senha segura
def senha_segura(senha):
    padrao = '^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$'
    return bool(re.match(padrao, senha))

print(senha_segura("Aa1@1234"))
print(senha_segura("senha123"))

#7. Extração de Palavras
