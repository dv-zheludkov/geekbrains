def my_func(arg1, arg2, arg3):
    my_list = sorted([arg1, arg2, arg3], reverse=True)
    return my_list[0] + my_list[1]

print(my_func(4, 3, 4))

