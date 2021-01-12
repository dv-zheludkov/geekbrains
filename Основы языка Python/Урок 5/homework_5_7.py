def make_dict(*args):
    dict_profit = {}
    dict_average_profit = {"Средняя прибыль": 0}
    list_average_profit = []
    for data_string in args:
        dohod = int(data_string[2])
        rashod = int(data_string[3])
        balans = dohod - rashod
        if balans > 0:
            list_average_profit.append(balans)
        dict_profit[data_string[0]] = balans
    dict_average_profit["Средняя прибыль"] = round(sum(list_average_profit)/len(list_average_profit), 2)
    import json
    data_to_json = json.dumps([dict_profit, dict_average_profit])
    with open("file7.json", "w") as file7json:
        file7json.writelines(data_to_json)


with open("file7.txt", "r") as file7:
    list_from_file = file7.readlines()
    data_from_file = list(map(str.split, list_from_file))
    make_dict(*data_from_file)
