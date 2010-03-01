/*
	 Author: Nicholas Mancuso
	 Implementation of the FireFly Algorithm to search for
	 the optimal solution in a two variable function.
	 Date: 2/4/10
 */

#include <unistd.h>
#include "firefly.h"

/*
    Our function declarations
*/


double our_func(ffly *fly);

int
main(int argc, char **argv)
{
    point p;
    double mins[] = { -5.0, -5.0 };
    double maxs[] = { 5.0, 5.0 };
    char c;
    size_t pop_count = POP_COUNT, max_gen = MAX_GEN;

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

    p = ffa(pop_count, max_gen, 2, mins, maxs, &our_func);

    printf("Max x: %.2lf, Max y %.2lf\n", p.x, p.y);

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




