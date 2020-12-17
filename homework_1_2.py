time = int(input('Введите время в секундах: '))
hours = time // 3600
min = time % 3600 // 60
sec = time % 60

print(f'Вы ввели время в секундах {time}. В формате чч:мм:сс это будет {hours:02}:{min:02}:{sec:02}.')