with open("file1.txt","w") as file1:
    data_string = " "
    data_in_file = []
    while data_string:
        data_string = input("Введите строку: ")
        data_in_file.append(data_string + "\n")
    file1.writelines(data_in_file)
