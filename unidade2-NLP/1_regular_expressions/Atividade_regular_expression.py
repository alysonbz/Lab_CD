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
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if ".." in email:
        return "email inválido"

    if re.fullmatch(regex, email):
        return "email válido"

    return "email inválido"

email = 'ricardo112.41@gmail..com'
print("resultado segunda questão: ", valid_email(email))

#3)

def achar_numero(text):
    regex = r"\(?\d{2}\)?\s?\d{4,5}-\d{4}|\b0800\s?\d{3}\s?\d{4}\b"
    numeros = re.findall(regex, text)

    return numeros

text = "A manhã de trabalho na agência começou bastante movimentada para Mariana. Logo às 9h, ela precisou entrar em contato com o suporte técnico para resolver um problema no servidor principal; ela discou rapidamente para o número (11) 4004-1234 e abriu o chamado de emergência.Assim que desligou, lembrou que precisava confirmar os detalhes do almoço de negócios com o novo fornecedor de embalagens, cujo contato comercial era (21) 98765-4321. Felizmente, a reunião foi confirmada para o meio-dia sem grandes imprevistos.No final da tarde, ao revisar a lista de pendências, Mariana percebeu que ainda faltava atualizar o cadastro da matriz. Sem perder tempo, ligou para a central de atendimento geral no 0800 770 5544 para validar os dados de faturamento. Com todas as pendências resolvidas e os números devidamente anotados na agenda, ela finalmente pôde encerrar o expediente com a sensação de dever cumprido."
print("resultado terceira questão: ",achar_numero(text))

#4)

def substituir_palavra(text):
    regex = r"\bgato\b"
    cachorro = re.sub(regex, "cachorro", text)
    return cachorro

text = "O gato está dormindo. Meu gato gosta de brincar."
print("resultado quarta questão:", substituir_palavra(text))

#5

def extrair_urls(text):
    regex = r"https?://[^\s]+"
    resp = re.findall(regex, text)
    return resp

text = "Acesse https://www.google.com para pesquisar."\
       "Também visite https://www.python.org para aprender Python."
print("resultado quinta questão:", extrair_urls(text))

#6

def verificar_senha(senha):
    regex = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$"

    if re.fullmatch(regex, senha):
        return "Senha segura"
    else:
        return "Senha não segura"

senha = "Python@123"
print("resultado sexta questão:", verificar_senha(senha))

#7

def extrair_palavras(text):
    regex = r"\b\w+\b"
    lista = re.findall(regex, text)
    return lista

text = "Python é uma linguagem de programação."
print("resultado sétima questão:", extrair_palavras(text))

#8

def validar_data(data):
    regex = r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$"

    if re.fullmatch(regex, data):
        return "Data válida"
    else:
        return "Data inválida"

data = "22/09/2026"
print("resultado oitava questão:", validar_data(data))

#9

def extrair_nomes(text):
    regex = r"\b[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][a-záàãâéêíóôõúç]+\b"
    lista = re.findall(regex, text)

    return lista

text = "Mariana foi visitar Eduardo e depois encontrou João."
print("resultado nona questão:", extrair_nomes(text))

#10

def contar_vogais(text):
    regex = r"[aeiouAEIOU]"
    lista = re.findall(regex, text)

    return len(lista)

text = "Python é uma linguagem de programação."
print("resultado décima questão:", contar_vogais(text))