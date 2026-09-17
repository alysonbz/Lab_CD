import re

def validate_date(date_str):
    regex = r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/\d{4}$"
    return bool(re.match(regex, date_str))

print(validate_date("25/12/2023"))
print(validate_date("32/13/2023"))