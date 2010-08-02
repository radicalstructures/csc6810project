from pso import *

def main():
    pso = PSO(10, 40, 2.0, 2.0)
    pso.run('dejung', 5)

if __name__=='__main__':
    main()
