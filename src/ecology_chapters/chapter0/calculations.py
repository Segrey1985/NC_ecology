import re

def add_calculated_placeholders(placeholders: dict) -> None:
    project_name_lower(placeholders)
    project_name_no_capacity(placeholders)
    project_name_genitive_lower(placeholders)


def project_name_lower(placeholders: dict) -> None:
    """Наименование проекта с маленькой буквы"""
    key = "НАИМЕНОВАНИЕ_ПРОЕКТА"
    new_key = "НАИМЕНОВАНИЕ_ПРОЕКТА_lower"
    value = placeholders[key]
    if not value:
        placeholders[new_key] = "{{" + str(new_key) + "}}"
        return
    placeholders[new_key] = str(value)[0].lower() + str(value)[1:]


def project_name_no_capacity(placeholders: dict) -> None:
    """Наименование проекта без мощности с маленькой буквы"""
    key = "НАИМЕНОВАНИЕ_ПРОЕКТА"
    new_key = "НАИМЕНОВАНИЕ_ПРОЕКТА_no_capacity"
    value = placeholders[key]
    if not value:
        placeholders[new_key] = "{{" + str(new_key) + "}}"
        return
    regex = r"мощностью\s+\d+(?:[.,]\d+)?\s+мвт"
    new_ = re.sub(regex, "", str(value), flags = re.IGNORECASE)
    new = re.sub(r"\s{2,}", " ", new_).strip()
    placeholders[new_key] = new[0].lower() + new[1:]
    

def project_name_genitive_lower(placeholders: dict) -> None:
    key = "name_genitive"
    new_key = "name_genitive_lower"
    value = placeholders[key]
    if not value:
        placeholders[new_key] = "{{" + str(new_key) + "}}"
        return
    placeholders[new_key] = str(value)[0].lower() + str(value)[1:]
