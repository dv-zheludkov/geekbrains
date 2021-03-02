class Worker:
    def __init__(self, name, surname, position, wage, bonus):
        self.name = name
        self.surname = surname
        self.position = position
        self._income = {"wage": wage,
                        "bonus": bonus}

class Position(Worker):
    def get_full_name(self):
        return f"{self.name} {self.surname}"
    def get_total_income(self):
        return (self._income["wage"] + self._income["bonus"])

person1 = Position("Adam", "Smith", "Cleaner", 2000, 400)
print(f"{person1.get_full_name()} has total income {person1.get_total_income()}")