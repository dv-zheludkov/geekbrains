def my_func1 (x, y):
    return x ** y

def my_func2 (x, y):
    step = abs(y)
    rez = 1
    while step > 0:
        rez *= x
        step -= 1
    return 1 / rez



print(my_func1(10, -3))
print(my_func2(10, -3))