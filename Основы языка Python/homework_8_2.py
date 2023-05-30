class MyZerroDivisionError(Exception):
    pass

    @staticmethod
    def my_division(arg1, arg2):
        try:
            if arg2 == 0:
                raise MyZerroDivisionError()
            else:
                return arg1 / arg2
        except MyZerroDivisionError:
            return "Деление на ноль. Недопустимая опреация"

arg1, arg2 = input("Введите делимое и делитель через пробел : ").split(" ")
print(MyZerroDivisionError.my_division(int(arg1), int(arg2)))