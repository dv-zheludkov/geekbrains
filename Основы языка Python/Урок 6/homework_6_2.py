class Road:
    def __init__(self, lenth, width, massa_1km_1cm = 25, sm = 5):
        self._lenth = lenth
        self._width = width
        self.__massa_1km_1cm = massa_1km_1cm
        self.__sm = sm
    def mass_as(self):
        return self._lenth * self._width * self.__massa_1km_1cm * self.__sm

road1 = Road(20, 5000)
print(road1.mass_as())
