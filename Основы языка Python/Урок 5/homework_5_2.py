with open("file2.txt", "r") as file2:
    data_from_file = file2.readlines()
    print("В контрольном файле ", len(data_from_file), " строк.")
    for num, el in enumerate(data_from_file):
        print("В строке ", num + 1 , " содежится", len(el.split(" ")), " слов.")