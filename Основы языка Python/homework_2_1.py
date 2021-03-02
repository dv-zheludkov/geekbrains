my_list = []
my_list.append(5)
my_list.append("Строка")
my_list.append(['с', 'п', 'и', 'с', 'о', 'к'])
my_list.append(list('еще список'))
for index,el in enumerate(my_list):
    print(f"{index}-й эелемент списка имеет тип {type(el)}")
