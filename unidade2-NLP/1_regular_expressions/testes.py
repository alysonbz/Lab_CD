from ntlk.corpus import stopwords
text = """The Cat is in the box. The Cat likes the box.
                   The box is over the Cat."""
tokens = [w for w in word_tokenize(text.lower())
                   if w.isalpha()]
print(Counter(no_stops).most_common())