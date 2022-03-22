def int_func(arg1):
    my_word = arg1
    return my_word[0].upper() + my_word[1:]

my_string = input("Введите слова, разделенные пробелом: ").split(" ")
new_string = ""
for el in my_string:
    new_string += int_func(el) + " "
new_string = new_string[:-1]
print(new_string)
