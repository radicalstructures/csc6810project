from firefly import *
from sys import argv

def main(args):
    f = Population(20, 40, 0.1, 1.0, 1.0)
    print 'running normal'
    f.run(args[1], int(args[2]), Population.NORMAL, int(args[3]), bool(args[4]))

    print f.pop[0]
    
    print 'running hybrid'
    f.run(args[1], int(args[2]), Population.HYBRID, int(args[3]), bool(args[4]))

    print f.pop[0]

if __name__ == '__main__':
    if len(argv) != 5:
        argv = ['run.py']
        argv.append('dejung')
        argv.append(3)
        argv.append(2)
        argv.append(False)

    argv[4] = argv[4].lower() == 'true'
    main(argv)
