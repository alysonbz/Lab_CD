import re

#1. Contagem de Correspondências
def cont_word(text):
    regex = "Python"
    resp = re.findall(regex,text)
    return len(resp)

texto1 = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."
print("Questão 1:", cont_word(texto1))


# 2. Validação de E-mail
def validar_email(email):
    regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(regex, email))

print("Questão 2:", validar_email("teste@email.com"))


# 3. Extração de Números de Telefone
def extrair_telefones(texto):
    regex = r"\(?\d{2}\)?\s?9?\d{4}-?\d{4}"
    return re.findall(regex, texto)

texto3 = "Meus contatos são (85) 99273-9317 e 85992739317."
print("Questão 3:", extrair_telefones(texto3))


# 4. Substituição de Palavras
def substituir_palavra(texto):
    regex = r"gato"
    return re.sub(regex, "cachorro", texto)

texto4 = "O gato subiu no telhado e o gato desceu correndo."
print("Questão 4:", substituir_palavra(texto4))


# 5. Extração de URLs
def extrair_urls(texto):
    regex = r"https?://[^\s]+"
    return re.findall(regex, texto)

texto5 = "Visite https://www.siteelyson.com ou http://google.com para saber mais."
print("Questão 5:", extrair_urls(texto5))


# 6. Verificação de Senha Segura
def senha_segura(senha):
    regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?\":{}|<>]).{8,}$"
    return bool(re.match(regex, senha))

print("Questão 6:", senha_segura("Senha@123"))
print("Questão 6 (fraca):", senha_segura("senha123"))


# 7. Extração de Palavras
def extrair_palavras(texto):
    regex = r"\b\w+\b"
    return re.findall(regex, texto)

texto7 = "Python é uma linguagem poderosa e versátil."
print("Questão 7:", extrair_palavras(texto7))


# 8. Validação de Data
def validar_data(data):
    regex = r"^\d{2}/\d{2}/\d{4}$"
    return bool(re.match(regex, data))

print("Questão 8:", validar_data("17/09/2026"))
print("Questão 8 (inválida):", validar_data("2026-09-17"))


# 9. Extração de Nomes Próprios
def extrair_nomes_proprios(texto):
    regex = r"\b[A-ZÀ-Ú][a-zà-ú]*\b"
    return re.findall(regex, texto)

texto9 = "Maria e João foram visitar Fortaleza no Ceará."
print("Questão 9:", extrair_nomes_proprios(texto9))


# 10. Contagem de Vogais
def contar_vogais(texto):
    regex = r"[aeiouAEIOU]"
    return len(re.findall(regex, texto))

print("Questão 10:", contar_vogais("Python é incrível"))