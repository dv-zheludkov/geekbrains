class MyExceptionNumbersOnly(Exception):
    pass

    @staticmethod
    def input_checker(num):
        try:
            if num.isdigit():
                return [True, int(num)]
            else:
                raise MyExceptionNumbersOnly
        except MyExceptionNumbersOnly:
            print("Вы ввели не число")
            return [False]

input_string = ""
my_list = []
while input_string != "stop":
    input_string = input("Введите число: ")
    if MyExceptionNumbersOnly.input_checker(input_string)[0]:
        my_list.append(MyExceptionNumbersOnly.input_checker(input_string)[1])

print(my_list)