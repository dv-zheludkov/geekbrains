def my_person (name, lastname, bd, city, email, cel):
    my_person_string = f"имя {name}, фамилия {lastname}, год рождения {bd}," \
                       f" город проживания {city}, email {email}, телефон {cel}"
    return my_person_string


print (my_person("Jon", "Dow", "unknown", "NY", "unknown", "unknown"))