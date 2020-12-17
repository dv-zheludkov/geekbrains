dohod = int(input('Введите значение выручки: '))
rashod = int(input('Введите значение издержек: '))
pribyl = dohod - rashod
if pribyl < 0:
    print("Компания работает в убыток.")
else:
    print("Компания ведет дела успешно.")
    print(f"Рентабельность выручки равна {(pribyl / dohod):.2f}.")
    chislo_sotr = int(input('Введите количество сотрудников: '))
    print(f"Прибыль фирмы в расчете на одного сотрудника  {(pribyl / chislo_sotr):.2f}.")

