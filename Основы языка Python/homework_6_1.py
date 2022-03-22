import time
from threading import Thread

class TrafficLight:
    def __init__(self):
        self.order = True
        self.__color = ""
        self.__timing = {"красный": 7,
                         "желтый": 2,
                         "зеленый": 30}
    def lights(self):
        check = False
        for el in self.__timing.keys():
            if check:
                yield el
            elif el == self.__color:
                check = True
                continue

    def set_color(self, new_color):
        self.__color = new_color
        print(f"Экстренно установлен {self.__color} цвет светофора. Введите новый цвет светофора: ")

    def running(self, color):
        self.__color = color
        while self.order:
            print(f"Установлен {self.__color} цвет светофора. Введите новый цвет светофора: ")
            time.sleep(self.__timing[self.__color])
            self.__color = next(self.lights(), "красный")


tl1 = TrafficLight()
thread_tl1 = Thread(target=tl1.running, args=("желтый", ))
thread_tl1.start()
while tl1.order:
    color = input()
    if color != next(tl1.lights(), "красный"):
        print("Нарушение последовательности переключения цветов. Светофор сломался и будет отключен")
        tl1.order = False
        thread_tl1.join()
    else:
        tl1.set_color(color)
