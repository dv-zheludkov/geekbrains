from itertools import count, cycle

def int_from_start(start, stop):
    if stop < start:
        return "Ошибка, условие остановки цикла должно быть больше начального значения!"
    else:
        for el in count(start):
            if el > stop:
                break
            else:
                print(el)


def repeat_from_list(list_for_cycle, stop):
    cnt = 0
    if stop < 1:
        return "Ошибка, условие остановки цикла должно быть больше 0!"
    else:
        for el in cycle(list_for_cycle):
            if cnt >= stop:
                break
            print(el)
            cnt += 1


int_from_start(3,7)
print()
repeat_from_list("qweerty", 13)


