import re

def extract_urls(text):
    regex = r"https?://[^\s]+"
    return re.findall(regex, text)

text = "Visite https://www.google.com e http://example.com para mais informações."
print(extract_urls(text))