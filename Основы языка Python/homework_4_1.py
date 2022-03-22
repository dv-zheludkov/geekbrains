from sys import argv

def zar_plata(*argv):
    vyrabotka = int(argv[0])
    stavka = int(argv[1])
    premiya = int(argv[2])
    oklad = vyrabotka * stavka
    dop_vyplata = oklad / 100 * premiya
    return oklad + dop_vyplata

real_argv = argv[1:]

print(zar_plata(*real_argv))