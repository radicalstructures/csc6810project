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


void our_func(ffly_population *pop);

int 
main(int argc, char **argv)
{		
    point p;
    point min, max;
    min.x = min.y = -5;
    max.x = max.y = 5;
    
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
    
    p = ffa(pop_count, max_gen, min, max, &our_func);
    
    printf("Max x: %.2lf, Max y %.2lf\n", p.x, p.y);
    
	return EXIT_SUCCESS;
};

void
our_func(ffly_population *pop)
{
    unsigned register int i = 0;
    double x, y, z;
    for (i = 0; i < pop->nfflies; i++)
    {
        x = pop->x_values[i];
        y = pop->y_values[i];
        
        z = exp(-((x - 4) * (x - 4)) - ((y - 4) * (y - 4))) +  
            exp(-((x + 4) * (x + 4)) - ((y - 4) * (y - 4))) + 
            (2 * (exp(-(x*x) - (y * y)) + exp(-(x*x) - ((y+4) * (y+4)) )));
        
        pop->light_values[i] = z;
    }
};




