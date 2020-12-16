time = int(input('Введите время в секундах: '))
sec = time % 60
min = time // 60
hours = 0
if min >= 60:
    hours = min // 60
    min = min % 60

print(f'Вы ввели время в секундах {time}. В формате чч:мм:сс это будет {hours:02}:{min:02}:{sec:02}.')