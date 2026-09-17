import re

def validade_email(email: str):
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    match = re.match(regex, email)
    return match != None

emails = {
    "valid": [
        "fulano@gmail.com",
        "sicrano@yahoo.com.br",
        "beltrano@hotmail.com",
        "doquinha@alu.ufc.br"
    ],
    "invalid": [
        "fulanogames.com",
        "sicrano+2#gmail.com",
        "Beltrano@hotmail",
        "doquinha%yahoo.com.br"
    ]
}

total_correct = 0
total_wrong = 0

for email in emails["valid"]:
    result = validade_email(email)
    if result == True: total_correct += 1
    else: total_wrong += 1
    print(f"Current Email: '{email}\nClassified Valid: {result}\nIs Validation Correct: {result == True}")
    print("-"*20)

for email in emails["invalid"]:
    result = validade_email(email)
    if result == False: total_correct += 1
    else: total_wrong += 1
    print(f"Current Email: '{email}\nClassified Valid: {result}\nIs Validation Correct: {result == False}")
    print("-"*20)

print(f"\nTotal Correct: {total_correct}\nTotal Wrong: {total_wrong}")