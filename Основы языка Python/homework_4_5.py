from functools import reduce

def my_multiply(arg1, arg2):
    return arg1 * arg2

origin_list = [el for el in range(100,1001) if el % 2 == 0]

rez = reduce(my_multiply, origin_list)

print(rez)