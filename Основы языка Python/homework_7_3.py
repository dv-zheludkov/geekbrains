class Kletka:
    def __init__(self, cell):
        self._cell = cell

    def __add__(self, other):
        new_cell = self._cell + other._cell
        return Kletka(new_cell)

    def __sub__(self, other):
        new_cell = self._cell - other._cell
        if new_cell > 0:
            return Kletka(new_cell)
        else:
            print("Вычитание невозможно, результат меньше 0")

    def __mul__(self, other):
        new_cell = self._cell * other._cell
        return Kletka(new_cell)

    def __truediv__(self, other):
        new_cell = self._cell // other._cell
        return Kletka(new_cell)

    def make_order(self, num):
        order = ""
        check = True
        order_cell = self._cell
        while check:
            if order_cell // num > 0:
                order += "*" * num + "\n"
                order_cell -= num
            else:
                order += "*" * order_cell
                check = False
        return order

kletka1 = Kletka(12)
kletka2 = Kletka(8)
kletka3 = kletka1 + kletka2
kletka4 = kletka1 - kletka2
kletka5 = kletka1 * kletka2
kletka6 = kletka1 / kletka2

print(kletka5.make_order(13))
