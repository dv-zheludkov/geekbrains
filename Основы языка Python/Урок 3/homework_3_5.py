def my_summ(*args):
    rez = 0
    for el in args:
        rez += el
    return rez

spec_sim = False
summ = 0
while not spec_sim:
    my_list = list(input("Введите числа разделенные пробелом, для окончания ввода введите запятую: ").split(" "))
    try:
        my_index = my_list.index(",")
        my_list = my_list[:my_index]
        spec_sim = True
    except ValueError:
        pass
    int_list = list(map(int, my_list))
    summ += my_summ(*int_list)
print(summ)