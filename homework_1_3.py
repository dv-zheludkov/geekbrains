digit = int(input('Введите число n в диапазоне от 0 до 9 включительно: '))
if digit >= 0 and digit <= 9:
    print("Сумма чисел n + nn + nnn равна: ",
          digit + int(str(digit) + str(digit)) + int(str(digit) + str(digit) + str(digit)))
else:
    print("Введенное число не входит в диапазоне от 0 до 9 включительно")
