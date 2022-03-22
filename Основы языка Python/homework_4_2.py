primer = [300, 2, 12, 44, 1, 1, 4, 10, 7, 1, 78, 123, 55]

new_list = [el for index, el in enumerate(primer) if el > primer[index - 1] and index > 0]

print(new_list)
