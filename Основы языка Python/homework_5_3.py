with open("file3.txt", "r") as file3:
    data_from_file = file3.readlines()
    sr_zarp = 0
    for data_string in data_from_file:
        el = data_string.split(" ")
        el[1] = int(el[1])
        if el[1] < 20000:
            print(el[0], "имеет доход менее 20000")
        sr_zarp += el[1]
    print("Средняя величина дохода - ", sr_zarp / len(data_from_file))