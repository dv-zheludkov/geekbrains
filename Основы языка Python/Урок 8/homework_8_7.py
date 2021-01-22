class  ComplexNumbers:
    def __init__(self, real, mean):
        self.real = real
        self.mean = mean

    def __str__(self):
        return f"{self.real}+{self.mean}i" if self.mean > 0 else f"{self.real}{self.mean}i"

    def __add__(self, other):
        return ComplexNumbers(self.real + other.real, self.mean + other.mean)

    def __mul__(self, other):
        real = self.real * other.real - self.mean * other.mean
        mean = self.mean * other.real + self.real * other.mean
        return ComplexNumbers(real, mean)


complex1 = ComplexNumbers(3,4)
complex2 = ComplexNumbers(2,-2)
complex3 = complex1 + complex2
complex4 = complex1 * complex2

print(f"Первое число {complex1}")
print(f"Второе число {complex2}")
print(f"Результат сложения {complex3}")
print(f"Результат умножения {complex4}")

