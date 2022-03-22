from random import randint
from functools import reduce

def sum_two_el(arg1, arg2):
    return arg1 + arg2

with open("file5.txt", "w") as file5:
    list_to_file = [str(randint(0, 100)) for el in range(100)]
    data_to_file = " ".join(list_to_file)
    file5.write(data_to_file)
with open("file5.txt", "r") as file5:
    data_from_file = list(map(int, file5.read().split(" ")))
    print("Сумма чисел в файле равна ", reduce(sum_two_el, data_from_file))


