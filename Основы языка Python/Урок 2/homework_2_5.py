my_list = [7, 5, 3, 3, 2]
el_new = int(input("Введите число: "))
index = 0
while True:
    if index == len(my_list):
        my_list.insert(index, el_new)
        break
    elif el_new > my_list[index]:
        my_list.insert(index,el_new)
        break
    index += 1
print(my_list)