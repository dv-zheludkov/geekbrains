class OrgStore:
    def __init__(self):
        self.store = []
        self.removed = []

    def add_in_store(self, arg):
        if isinstance(arg, (Printer, Scaner, Copir)):
            self.store.append(arg)
        else:
            print("Объект не добавлен, так как не является объектом классов Принтер, Сканер или Копир")

    def move_from_store(self, arg, move_to):
        self.store.remove(arg)
        arg.source = move_to
        self.removed.append(arg)

    def list_type(self):
        my_dict = {"Принтеры": 0,
                   "Сканеры": 0,
                   "Копиры": 0}
        for el in self.store:
            if isinstance(el, Printer):
                my_dict["Принтеры"] +=1
            if isinstance(el, Scaner):
                my_dict["Сканеры"] +=1
            if isinstance(el, Copir):
                my_dict["Копиры"] +=1
        return my_dict

    def list_all_store(self):
        for el in self.store:
            print(str(el))

    def list_all_removed(self):
        for el in self.removed:
            print(str(el))

class OrgTechnika:
    def __init__(self, source, name, inv_n, aformat, iscolor = True, isnet=True):
        self.source = source
        self.name = name
        self.inv_n = inv_n
        self.aformat = aformat
        self.iscolor = iscolor
        self.isnet = isnet

    def __str__(self):
        class_out = ""
        class_out += self.source + " "
        class_out += self.name + " "
        class_out += self.inv_n + " "
        class_out += self.aformat + " "
        class_out += "Цветной " if self.iscolor else "Черно-белый "
        class_out += "Сетевой " if self.isnet else "Не имеет сетевого разъема "
        return class_out

class Printer(OrgTechnika):
    def __init__(self, source, name, inv_n, aformat,  type, iscolor = True, isnet=True):
        self.type = type
        super().__init__(source, name, inv_n, aformat, iscolor, isnet)

    def __str__(self):
        class_out = super().__str__()
        class_out += self.type + " "
        return class_out

class Scaner(OrgTechnika):
    def __init__(self, source, name, inv_n, aformat, res, iscolor = True, isnet=True):
        self.res = res
        super().__init__(source, name, inv_n, aformat, iscolor, isnet)

    def __str__(self):
        class_out = super().__str__()
        class_out += self.res + " "
        return class_out

class Copir(OrgTechnika):
    def __init__(self, source, type, name, inv_n, aformat, iscolor = True, isnet=True):
        self.type = type
        super().__init__(source, name, inv_n, aformat, iscolor, isnet)

    def __str__(self):
        class_out = super().__str__()
        class_out += self.type + " "
        return class_out

printer1 = Printer("Новый", "Xerox 34555", "222", "A4", "Лазерный", iscolor=True, isnet=False)
printer2 = Printer("Бухгалтерия", "Canon 22", "225", "A4", "Струйный", isnet=False)
scaner1 = Scaner("Бухгалтерия", "Canon 3", "245", "A4", "1600x1200")

sklad = OrgStore()
sklad.add_in_store(printer1)
sklad.add_in_store(printer2)
sklad.add_in_store(scaner1)
sklad.add_in_store("Не будет добавлен, так как не пройдет валидацию")
print()
print(("Выводим список всего на складе"))
print(sklad.list_type())
sklad.list_all_store()
print()
print("Перемещаем принтер в отдел кадров. При этом сам объект \n"
      " помещаем в спецхранилище перемещенных объектов, чтобы знать куда ушла позиция\n"
      "и опять выводим содержимое склада")
sklad.move_from_store(printer2, "Отдел кадров")
print()
print(sklad.list_type())
sklad.list_all_store()
print()
print("Выводим список перемещенных со склада единиц оргтехники")
sklad.list_all_removed()