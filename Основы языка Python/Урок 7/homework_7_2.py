class Clothes:
    def __init__(self, name="-"):
        self.name = name
        self.rashod_tkani

class Coat(Clothes):
    def __init__(self, razmer, name="-"):
        self.razmer = razmer
        super().__init__(name)
    @property
    def rashod_tkani(self):
        return self.razmer / 6.5 + 0.5

class Suit(Clothes):
    def __init__(self, rost, name="-"):
        self.rost = rost
        super().__init__(name)
    @property
    def rashod_tkani(self):
        return self.rost * 2 + 0.3

coat1 = Coat(50)
coat2 = Coat(52)
coat3 = Coat(56)
suit1 = Suit(3)
suit2 = Suit(5, name = "Вечерний костюм")
#Производим расчет расхода ткани для производства 50 экземпляров каждого вида одежды
proizvodstvo = 50 * coat1.rashod_tkani + 50 * coat2.rashod_tkani +\
    50 * coat3.rashod_tkani + 50 * suit1.rashod_tkani + 50 * suit2.rashod_tkani

print(f"{proizvodstvo:.2f}")