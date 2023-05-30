def fact(n):
    fct = 1
    for num in [el for el in range(1,n+1)]:
        fct *= num
        yield fct


for el in fact(4):
    print(el)