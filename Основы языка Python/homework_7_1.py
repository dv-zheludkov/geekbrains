class Matrix:

    def __init__(self, *args):
        self.__iner_natrix = []
        self.width = len(args[0])
        self.height = len(args)
        for el in args:
            self.__iner_natrix.append(el)

    def __str__(self):
        matrix_out = ""
        for el in self.__iner_natrix:
            str_out = ""
            for cel in el:
                str_out += f"{cel:4}"
            matrix_out += f"{str_out}\n"
        return matrix_out

    def __add__(self, other):
        new_matrix = []
        height = self.height
        while height > 0:
            width = self.width
            new_str_m = []
            while width > 0:
                new_str_m.append(self.__iner_natrix[self.height - height][self.width - width] +
                                 other.__iner_natrix[other.height - height][other.width - width])
                width -= 1
            new_matrix.append(new_str_m)
            height -= 1
        return Matrix(*new_matrix)

matrix1 = Matrix([1, 2, 3], [4, 5, 6], [2, 1, 9], [7, 8, 9])
matrix2 = Matrix([5, 4, 3], [7, 1, 6], [8, 8, 3], [7, 5, 4])

matrix3 = matrix1 + matrix2

print(matrix1)
print(matrix2)
print(matrix3)
