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

# 2. Validação de e-mail

def cont_mail(mail):
    regex = r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}"
    resultado = re.search(regex,mail)
    return resultado.group() if resultado else "Nenhum e-mail encontrado"

mail = "Meu e-mail é dayanemagalhaes@alu.ufc.br"
print(cont_mail(mail))

# 3. Extração de números de telefones

def cont_contato(numero):
    regex = r"\(\d{2}\)\s?\d{9}"
    resultado = re.search(regex,numero)
    return resultado.group() if resultado else "Nenhum número de telefone encontrado"

contato = "Meu e-mail é dayanemagalhaes@alu.ufc.br e meu número de contato é (85) 999299229"
print(cont_contato(contato))

# 4. Substitução de palavras

def substituir_palavra(texto):
    regex = r"gato"
    resultado = re.sub(regex, "cachorro", texto)
    return resultado

texto_original = "O gato subiu no telhado, e depois o outro gato miou."
print(substituir_palavra(texto_original))

# 5. Extração de URLs

def extrair_urls(texto):
    regex = r"https?://[^\s]+"
    resultado = re.findall(regex, texto)
    return resultado if resultado else "Nenhuma URL encontrada"

texto_exemplo = "Acesse o buscador em https://google.com ou veja nosso código em http://github.com."
print(extrair_urls(texto_exemplo))

# 6. Verificação de senha de segurança

def verificar_senha_segura(senha):
    regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    resultado = re.match(regex, senha)
    return "Senha segura" if resultado else "Senha insegura"

print(verificar_senha_segura("Senha@123"))
print(verificar_senha_segura("12345"))

# 7. Extração de Palavras

def extrair_palavras(texto):
    regex = r"\w+"
    resultado = re.findall(regex, texto)
    return resultado if resultado else "Nenhuma palavra encontrada"

texto_exemplo = "Olá, mundo! Programar em Python é muito divertido."
print(extrair_palavras(texto_exemplo))

# 8. Validação de Data

def validar_data(data):
    regex = r"^\d{2}/\d{2}/\d{4}$"
    resultado = re.match(regex, data)
    return "Data válida" if resultado else "Data inválida"

print(validar_data("11/09/2026"))
print(validar_data("11-09-26"))

# 9. Extração de Nomes Próprios

def extrair_nomes_proprios(texto):
    regex = r"\b[A-ZÀ-Ú][a-zà-ú]+"
    resultado = re.findall(regex, texto)
    return resultado if resultado else "Nenhum nome próprio encontrado"

texto_exemplo = "O aluno Carlos estuda na UFC com a professora Maria."
print(extrair_nomes_proprios(texto_exemplo))

# 10. Contagem de Vogais

def contar_vogais(texto):
    regex = r"[aeiouAEIOUáéíóúÁÉÍÓÚâêîôûÂÊÎÔÛãõÃÕ]"
    resultado = re.findall(regex, texto)
    return len(resultado)

texto_exemplo = "Atividade de Expressões Regulares."
print(f"Total de vogais: {contar_vogais(texto_exemplo)}")