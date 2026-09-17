import re

string = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro. 2026"

match1 = re.match(r"\w+", string)
print("Match 1:", match1.group() if match1 else "No match")

match2 = re.match(r"\d", string)
print("Match 2:", match2.group() if match2 else "No match")

match3 = re.match(r"\s", string)
print("Match 3:", match3.group() if match3 else "No match")

match4 = re.match(r".*", string)
print("Match 4:", match4.group() if match4 else "No match")

match5 = re.match(r".a*", string)
print("Match 5:", match5.group() if match5 else "No match")

match6 = re.findall(r"\d+", string)
print("Match 6:", match6 if match6 else "No match")

match7 = re.split(r"\s+", string)
print("Match 7:", match7 if match7 else "No match")

match8 = re.split(r"\.", string)
print("Match 8:", match8 if match8 else "No match")

match9 = re.split(r"[a-z]", string)
print("Match 9:", match9 if match9 else "No match")

match10 = re.match(r"[a-z]\w+", string)
print("Match 10:", match10.group() if match10 else "No match")

match11 = re.findall(r"[a-z]\w+", string)
print("Match 11:", match11 if match11 else "No match")

match12 = re.search(r"[a-z]\w+", string)
print("Match 12:", match12.group() if match12 else "No match")