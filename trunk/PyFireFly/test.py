from firefly import *
from sys import argv

def main(args):
    f = Population(10, 40, 0.1, 1.0, 1.0)
    print 'testing normal'
    f.test(args[1], int(args[2]), Population.NORMAL, int(args[3]), bool(args[4]))

    print f.pop[0]
    
    print 'testing hybrid'
    f.test(args[1], int(args[2]), Population.HYBRID, int(args[3]), bool(args[4]))

    print f.pop[0]

if __name__ == '__main__':
    if len(argv) != 5:
        argv = ['test.py']
        argv.append('dejung')
        argv.append(3)
        argv.append(2)
        argv.append(False)

    argv[3] = argv[3].lower() == 'true'
    main(argv)
