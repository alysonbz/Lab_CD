import re

def extract_proper_names(text):
    regex = r"\b[A-Z][a-zá-ú]*\b"
    return re.findall(regex, text)

text = "Maria e João viajaram para São Paulo com Pedro."
print(extract_proper_names(text))