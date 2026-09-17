import re

#1. **Contagem de Correspondências:**
#   Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões regulares.
print("===== QUESTÃO 1 =====")

def cont_word(text):
    regex = "Python"
    resp = re.findall(regex,text)
    return len(resp)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print("resultado primeira questão: ",cont_word(text))

# ====== QUESTÃO 2 - validação de E-mail =====
print("\n===== QUESTÃO 2 =====")

def validar_email(email):
    regex = r"^[a-z A-Z 0-9 _.+-]+@[a-z A-Z 0-9]+\.[a-z A-Z 0-9-.]+$"
    if re.match(regex, email):
        return "E-mail válido!"
    else:
        return "E-mail inválido!"

email = "usuario@gmail.com"

print(validar_email(email))

# ==== QUESTÃO 3 - Extração de números de telefone ====
print("\n===== QUESTÃO 3 =====")

regex = r"\d{2}\d{5}-\d{4}"
num_telefone = "Ligue para o número 8599111-1111"
val_telefone = re.findall(regex, num_telefone)

if val_telefone:
    print(f"Telefone encontrado: {val_telefone}")
    print("Telefone Válido!")
else:
    print("Nenhum telefone encontrado!")

# === Questão 4 - Substituição de palavras ====
print("\n===== QUESTÃO 4 =====")

def subtituir(texto):
    partes = re.split(r"gato", texto)
    novo_texto = "cachorro".join(partes)
    return novo_texto

texto1 = "Meu gato está doente."
resultado = subtituir(texto1)
print(resultado)

# ==== QUESTÃO 5 - extração de URLs ====
print("\n===== QUESTÃO 5 =====")

site = "Visite nosso site https://www.exemplo.com e também https://site.com.br/cursos"

urls = re.findall(r"https?://[^\s]+", site)
print(f"URLs encontradas: {urls}")

# ===== QUESTÃO 6 - verificação de segurança =====
print("\n===== QUESTÃO 6 =====")

def verifica_senha(senha):
    if len(senha) < 8:
        return "Insegura: Menos de 8 caracteres."
    if not re.search(r"[A-Z]", senha):
        return "Insegura: Faltam letras maiúsculas."
    if not re.search(r"[a-z]", senha):
        return "Insegura: Faltam letras minúsculas."
    if not re.search(r"\d", senha):
        return "Insegura: Faltam números."
    return "Senha Segura!"

print(verifica_senha("SenhaFraca1"))


# ===== QUESTÃO 7 - extração de palavras =====
print("\n===== QUESTÃO 7 =====")

texto2 = "Aprendendo expressões regulares em Python."
palavras = re.findall(r"\w+", texto2)

print(palavras)


# ===== QUESTÃO 8 - validação de data =====
print("\n===== QUESTÃO 8 =====")

def valida_data(data):
    padrao = r"^\d{2}/\d{2}/\d{4}$"
    if re.match(padrao, data):
        return "Data válida!"
    return "Data inválida!"

print(valida_data("25/12/2026"))


# ===== QUESTÃO 9 - extração de nomes próprios =====
print("\n===== QUESTÃO 9 =====")

texto3 = "Maria foi com João para Fortaleza comprar presentes."
# \b limita a palavra, [A-Z] letra maiúscula inicial seguida de minúsculas
nomes = re.findall(r"\b[A-Z][a-z]+\b", texto3)

print(nomes)


# ===== QUESTÃO 10 - contagem de vogais =====
print("\n===== QUESTÃO 10 =====")

texto4 = "Python é uma linguagem de programação."
# Colocamos todas as vogais (com e sem acento) dentro de colchetes
vogais = re.findall(r"[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]", texto4)

print(f"Total de vogais encontradas: {len(vogais)}")
