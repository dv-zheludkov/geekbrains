kol = int(input("Введите количество товаров: "))
tovary = []
count = 1
while count <= kol:
    stroka = list(input(f"Введите через запятую название {count}-го товара,"
                       " цену, количество, единицу измерения: ").split(","))
    tovar_dict = {"Название": stroka[0],
                  "Цена": stroka[1],
                  "Количество": stroka[2],
                  "Единица измерения": stroka[3]}
    tovar = [count,tovar_dict]
    tovary.append(tuple(tovar))
    count += 1

nazvanie_list = []
tsena_list = []
kolichestvo = []
ed_list = []
for el in tovary:
    nazvanie_list.append(el[1]["Название"])
    tsena_list.append(el[1]["Цена"])
    kolichestvo.append(el[1]["Количество"])
    ed_list.append(el[1]["Единица измерения"])
tovary_dict = {"Название": list(set(nazvanie_list)),
               "Цена": list(set(tsena_list)),
               "Количество": list(set(kolichestvo)),
               "Единица измерения": list(set(ed_list))}
print(tovary_dict)
