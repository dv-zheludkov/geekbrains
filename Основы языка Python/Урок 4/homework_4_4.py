from collections import Counter

primer = [2, 2, 2, 7, 23, 1, 44, 44, 3, 2, 10, 7, 4, 11]
counter = Counter(primer)
my_list = [el for el, n in counter.items() if n == 1]

print(my_list)