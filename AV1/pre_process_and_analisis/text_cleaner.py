import re
import unicodedata

def worker_clean_chunk(chunk: list) -> list:
    """Worker para limpeza básica e tratamento estrutural."""
    cleaned_list = []
    for text in chunk:
        if not isinstance(text, str):
            cleaned_list.append("")
            continue
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = "".join(c for c in text if unicodedata.category(c) != 'Mn')
        text = re.sub(r'o que gostei:', 'gostei', text)
        text = re.sub(r'o que nao gostei:', 'nao_gostei', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_list.append(text)
    return cleaned_list