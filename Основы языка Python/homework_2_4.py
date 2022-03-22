my_list = list(input("Введите строку из нескольких слов :").split(" "))
for index, el in enumerate(my_list):
    print(f"Строка {index +1} - {el[:10]}")
