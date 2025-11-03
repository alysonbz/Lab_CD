# ============================================
# 🧠 Atividade: Expressões Regulares em Python
# ============================================

import re

# 1️⃣ Contagem de Correspondências
def contar_python(texto):
    return len(re.findall(r'\bPython\b', texto, re.IGNORECASE))

# Exemplo:
print("1) Contagem de 'Python':", contar_python("Python é legal. Eu estudo python todo dia."))


# 2️⃣ Validação de E-mail
def validar_email(email):
    padrao = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(padrao, email))

print("2) Validação de e-mail:", validar_email("exemplo@gmail.com"))


# 3️⃣ Extração de Números de Telefone
def extrair_telefones(texto):
    padrao = r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}'
    return re.findall(padrao, texto)

print("3) Telefones:", extrair_telefones("Me ligue em (85)99876-1234 ou 8591234567."))


# 4️⃣ Substituição de Palavras
def substituir_gato_por_cachorro(texto):
    return re.sub(r'\bgato\b', 'cachorro', texto, flags=re.IGNORECASE)

print("4) Substituição:", substituir_gato_por_cachorro("O gato dorme. Gato é esperto."))


# 5️⃣ Extração de URLs
def extrair_urls(texto):
    padrao = r'https?://[^\s]+'
    return re.findall(padrao, texto)

print("5) URLs:", extrair_urls("Acesse https://openai.com e http://google.com agora!"))


# 6️⃣ Verificação de Senha Segura
def senha_segura(senha):
    padrao = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return bool(re.match(padrao, senha))

print("6) Senha segura:", senha_segura("SenhaForte@123"))


# 7️⃣ Extração de Palavras
def extrair_palavras(texto):
    return re.findall(r'\b\w+\b', texto)

print("7) Palavras:", extrair_palavras("Python é uma linguagem poderosa!"))


# 8️⃣ Validação de Data
def validar_data(data):
    padrao = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/[0-9]{4}$'
    return bool(re.match(padrao, data))

print("8) Data válida:", validar_data("31/12/2025"))


# 9️⃣ Extração de Nomes Próprios
def extrair_nomes_proprios(texto):
    return re.findall(r'\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\b', texto)

print("9) Nomes próprios:", extrair_nomes_proprios("Maria e João foram ao Ceará visitar Ana."))


# 🔟 Contagem de Vogais
def contar_vogais(texto):
    return len(re.findall(r'[aeiouAEIOU]', texto))

print("10) Vogais:", contar_vogais("Expressões regulares são úteis."))
