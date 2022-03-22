dic_en_ru = {"One": "Один",
             "Two": "Два",
             "Three": "Три",
             "Four": "Четыре"}

with open("file4.txt", "r") as file4:
    data_from_file = file4.readlines()
    data_to_file = []
    for el in data_from_file:
        target_string = el.split(" ")[0]
        data_to_file.append(el.replace(target_string, dic_en_ru[target_string]))
    with open("file4ru.txt", "w") as file4ru:
        file4ru.writelines(data_to_file)