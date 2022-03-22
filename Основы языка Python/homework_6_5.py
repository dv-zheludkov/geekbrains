class Stationery:
    def __init__(self, title):
        self.title = title
    def draw(self):
        print("Запуск отрисовки")

class Pen(Stationery):
    def draw(self):
        print(f"Запуск  отрисовки класса Pen, объект {self.title}")

class Pencil(Stationery):
    def draw(self):
        print(f"Запуск отрисовки класса Pencil, объект {self.title}")

class Handle(Stationery):
    def draw(self):
        print(f"Запуск отрисовки класса Handle, объект {self.title}")

pen1 = Pen("Старая ручка")
pencil1 = Pencil("Модный Parker")
handle1 = Handle("Оранжевый марке")

pen1.draw()
pencil1.draw()
handle1.draw()