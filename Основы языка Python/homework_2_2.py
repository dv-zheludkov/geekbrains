my_list = list(input("Введите элементы списка через запятую без пробелов: ").split(","))
len_my_list = len(my_list)
if len_my_list < 2:
    print("Здесь нечего переставлять")
else:
    while len_my_list > 1:
        my_list[len(my_list) - len_my_list], my_list[len(my_list) - len_my_list + 1] = \
            my_list[len(my_list) - len_my_list + 1], my_list[len(my_list) - len_my_list]
        len_my_list -= 2
print(my_list)
