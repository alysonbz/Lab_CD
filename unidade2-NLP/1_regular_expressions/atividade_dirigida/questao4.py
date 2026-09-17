import re

def replace_words(text):
    return re.sub(r"\bgato\b", "cachorro", text, flags=re.IGNORECASE)

text = "O gato subiu no muro e outro gato miou."
print(replace_words(text))