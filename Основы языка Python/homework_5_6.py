def dict_from_file(*args):
    dict_for_print = {}
    for el in args:
        import re
        str_num = el[1].split(" ")
        data_num = 0
        for num in str_num:
            if any(map(str.isdigit, num)):
                nums = re.sub("[^0-9]", "", num)
            else:
                nums = "0"
            data_num += int(nums)
        dict_for_print[el[0]] = data_num
    print(dict_for_print)


with open("file6.txt", "r") as file6:
    data_from_file = file6.readlines()
    list_to_dict = []
    for el in data_from_file:
        list_to_dict.append(el.split(": "))
    dict_from_file(*list_to_dict)


