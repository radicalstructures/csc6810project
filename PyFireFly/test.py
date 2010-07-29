from firefly import *
from sys import argv

def main(args):
    f = Population(20, 40, 0.1, 1.0, 1.0)
    for _ in range(int(args[4])):
        f.test(args[1], int(args[2]), Population.NORMAL, int(args[3]), bool(args[5]))

        regval = f.pop[0].val

        f.test(args[1], int(args[2]), Population.HYBRID, int(args[3]), bool(args[5]))

        print regval, ',', f.pop[0].val

if __name__ == '__main__':
    if len(argv) != 6:
        argv = ['run.py']
        argv.append('dejung')
        argv.append(3)
        argv.append(2)
        argv.append(1)
        argv.append(False)

    argv[5] = argv[5].lower() == 'true'
    main(argv)
