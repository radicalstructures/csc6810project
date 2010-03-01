/*
	 Author: Nicholas Mancuso
	 Implementation of the FireFly Algorithm to search for
	 the optimal solution in a two variable function.
	 Date: 2/4/10
 */

#include <unistd.h>
#include "firefly.h"

#define PI 3.14159265
#define E 2.71828183

#define N_ACKLEY 128
#define F_ACKLEY 128.0
/*
    Our function declarations
*/


double our_func(ffly *fly);
double akley(ffly *fly);

int
main(int argc, char **argv)
{
    double *mins = (double*)calloc(N_ACKLEY, sizeof(double));
    double *maxs = (double*)calloc(N_ACKLEY, sizeof(double));
    
    char c;
    size_t pop_count = POP_COUNT, max_gen = MAX_GEN;
    size_t i = 0;
    
    for (i=0; i < N_ACKLEY; i++)
    {
        mins[i] = -32.768;
        maxs[i] = 32.768;
    }
    
    while ( (c = getopt(argc, argv, "n:g:")) != -1)
    {
        switch (c)
        {
        case 'n':
            pop_count = atoi(optarg);
            break;
        case 'g':
            max_gen = atoi(optarg);
            break;
        case '?':
            return EXIT_FAILURE;
            break;
        default:
            printf("FFlies usage: -n NumberOfFlies -g NumberOfGenerations\n");
            break;
        }
    }

    ffa(pop_count, max_gen, N_ACKLEY, mins, maxs, &akley);
    free(mins);
    free(maxs);
    
    return EXIT_SUCCESS;
};

double
our_func(ffly *fly)
{   
    double x, y, z;
    
    x = fly->params[0];
    y = fly->params[1];

    z = exp(-((x - 4) * (x - 4)) - ((y - 4) * (y - 4))) +
        exp(-((x + 4) * (x + 4)) - ((y - 4) * (y - 4))) +
        (2 * (exp(-(x*x) - (y * y)) + exp(-(x*x) - ((y+4) * (y+4)) )));

    return z;
};



double
akley(ffly *fly)
{
    unsigned register int i = 0;
    
    double sumsq = 0.0, sumcos = 0.0;
    for (i = 0; i < N_ACKLEY; i++)
    {
        sumsq += fly->params[i] * fly->params[i];
        sumcos += cos(2 * PI * fly->params[i]);
    }
    return 20 * exp(-0.2*sqrt((1.0/F_ACKLEY) * sumsq)) - 
            exp((1.0/F_ACKLEY)*sumcos) + 20 + E;
};
        



