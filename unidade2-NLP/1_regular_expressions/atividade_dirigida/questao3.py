import re

def extract_phone_numbers(text):
    regex = r"\(\+\d+\)\d+"
    match = re.findall(regex, text)
    return match if match else "No Match"

text = "os números dos meus pais são: (+55)85991987623, (+55)85123456789"

print(f"Phone Numbers: {extract_phone_numbers(text)}")