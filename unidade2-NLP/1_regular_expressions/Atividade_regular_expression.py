# %% [markdown]
# # Atividade: Regular Expressions (Python)
# Cada célula abaixo resolve uma questão da atividade.

# %%
import re

# %% [markdown]
# ## 1. Contagem de Correspondências
# Conta quantas vezes a palavra "Python" aparece em uma string.

# %%
def contar_python(texto):
    return len(re.findall(r'\bPython\b', texto))

texto1 = "Python é ótimo. Aprender Python é divertido, e Python está em todo lugar. Pythonista não conta."
print("Questão 1 -> 'Python' aparece", contar_python(texto1), "vezes")

# %% [markdown]
# ## 2. Validação de E-mail

# %%
def validar_email(email):
    padrao = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'
    return re.fullmatch(padrao, email) is not None

for e in ["joao.silva@gmail.com", "maria@ufc.edu.br", "usuario@dominio", "@semnome.com", "texto qualquer", "a@b.c"]:
    print(f"Questão 2 -> {e!r:28} válido? {validar_email(e)}")

# %% [markdown]
# ## 3. Extração de Números de Telefone
# Aceita formatos brasileiros como `(85) 99999-1234`, `85 3222-1234`, `+55 88 98888-7777` e `3333-4444`.

# %%
def extrair_telefones(texto):
    padrao = r'(?<!\d)(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\d{4}|\d{4})-?\d{4}(?!\d)'
    return re.findall(padrao, texto)

texto3 = """Contatos: (85) 99999-1234, 85 3222-1234, +55 88 98888-7777,
fixo 3333-4444 e celular 91234-5678. Ano: 2024 (não é telefone)."""
print("Questão 3 -> telefones:", extrair_telefones(texto3))

# %% [markdown]
# ## 4. Substituição de Palavras
# Troca "gato" por "cachorro".

# %%
def substituir_gato(texto):
    return re.sub(r'\bgato\b', 'cachorro', texto)

texto4 = "O gato subiu no telhado. Meu gato gosta de leite, mas o gatorade não muda."
print("Questão 4 ->", substituir_gato(texto4))

# %% [markdown]
# ## 5. Extração de URLs

# %%
def extrair_urls(texto):
    padrao = r'(?:https?://|www\.)[^\s<>"\']*[^\s<>"\'.,;:!?)]'
    return re.findall(padrao, texto)

texto5 = ("Acesse https://www.ufc.br, veja também http://exemplo.com/pagina?id=10 "
          "e www.python.org. Mais em https://github.com/usuario/repo!")
print("Questão 5 -> URLs:", extrair_urls(texto5))

# %% [markdown]
# ## 6. Verificação de Senha Segura
# Mínimo de 8 caracteres, com maiúscula, minúscula, número e caractere especial.

# %%
def senha_segura(senha):
    padrao = r'(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9\s]).{8,}'
    return re.fullmatch(padrao, senha) is not None

for s in ["Abc@1234", "abc12345", "ABCDEFG@1", "Ab@1", "Senha#Forte99", "SenhaSemNumero@"]:
    print(f"Questão 6 -> {s!r:18} segura? {senha_segura(s)}")

# %% [markdown]
# ## 7. Extração de Palavras

# %%
def extrair_palavras(texto):
    return re.findall(r'[^\W\d_]+', texto)   # só letras (inclui acentos)

texto7 = "Olá, mundo! Regex é útil: aprenda em 2024, rápido."
print("Questão 7 -> palavras:", extrair_palavras(texto7))

# %% [markdown]
# ## 8. Validação de Data (dd/mm/aaaa)
# A regex valida o **formato** (dia 01-31, mês 01-12, ano com 4 dígitos).

# %%
def validar_data(data):
    padrao = r'(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}'
    return re.fullmatch(padrao, data) is not None

for d in ["25/12/2024", "01/01/2000", "32/01/2024", "10/13/2024", "5/5/2024", "2024/12/25", "31/04/2024"]:
    print(f"Questão 8 -> {d:12} formato válido? {validar_data(d)}")
print("Obs.: '31/04/2024' passa no formato, mas abril não tem dia 31 (a regex valida só o formato).")

# %% [markdown]
# ## 9. Extração de Nomes Próprios
# Palavras iniciadas por letra maiúscula (inclui a primeira palavra de cada frase).

# %%
def extrair_nomes_proprios(texto):
    padrao = r'\b[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúüç]+\b'
    return re.findall(padrao, texto)

texto9 = "Ontem Maria e João viajaram para Fortaleza. Depois visitaram Russas com a Ana Beatriz."
print("Questão 9 -> nomes próprios:", extrair_nomes_proprios(texto9))

# %% [markdown]
# ## 10. Contagem de Vogais

# %%
def contar_vogais(texto):
    return len(re.findall(r'[aeiouáàâãéêíóôõúü]', texto, flags=re.IGNORECASE))

texto10 = "Expressões Regulares são poderosas!"
print(f"Questão 10 -> '{texto10}' tem", contar_vogais(texto10), "vogais")