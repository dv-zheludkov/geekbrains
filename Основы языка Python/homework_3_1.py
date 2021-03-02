def my_div (arg1, arg2):
    arg1 = int(arg1)
    arg2 = int(arg2)
    try:
        rez = arg1 / arg2
    except ZeroDivisionError:
        rez = "На ноль делить нельзя!"
    return rez

arg1, arg2 = input("Введите два числа через пробел: ").split(" ")

print(my_div(arg1, arg2))