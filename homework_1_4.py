digit = int(input('Введите целое положительно число: '))
if digit < 0:
    print("Введенное число не является целым положительным числом")
else:
    max = 0
    while digit > 0:
        may_be_max = digit % 10
        if max < may_be_max:
            max = may_be_max
        digit = digit // 10
    print(f"Самая большая цифра в числе это {max}.")
