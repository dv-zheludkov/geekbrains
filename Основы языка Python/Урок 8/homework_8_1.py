class Data:
    def __init__(self, string):
        self.string = string

    @classmethod
    def Extractor(cls, string):
        d, m, y = string.split("-")
        return [int(d), int(m), int(y)]

    @staticmethod
    def Validator(args):
        val_report = ""
        if args[0] in range(1, 32):
            val_report += "Количество дней валидно\n"
        else:
            val_report += "Количество дней не валидно\n"
        if args[1] in range(1, 13):
            val_report += "Количество месяцев валидно\n"
        else:
            val_report += "Количество месяцев не валидно\n"
        if args[2] in range(1, 2022):
            val_report += "Количество лет валидно\n"
        else:
            val_report += "Количество лет не валидно\n"
        return val_report

data1 = Data("22-01-2021")
data2 = Data("32-13-2035")

print(data1.string)
print(Data.Validator(Data.Extractor(data1.string)))
print(data2.string)
print(Data.Validator(Data.Extractor(data2.string)))