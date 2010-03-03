/*
	 Author: Nicholas Mancuso
	 Implementation of the FireFly Algorithm to search for
	 the optimal solution in an n-variable function.
	 Date: 2/4/10
 */

#include <unistd.h>
#include "firefly.h"

#define PI 3.14159265
#define E 2.71828183

#define POP_COUNT 50
#define MAX_GEN 50

/*
    Our function declarations
*/
double yang(const ffly *fly, const size_t nparams);
double akley(const ffly *fly, const size_t nparams);
double schwefel(const ffly *fly, const size_t nparams);
double rosenbrock(const ffly *fly, const size_t nparams);

int
main(int argc, char **argv)
{
    double *mins;
    double *maxs;
	double min=0.0, max=0.0;
    char c;
    size_t pop_count = POP_COUNT, max_gen = MAX_GEN;
    size_t i = 0, dimension = 2;
    size_t nffa = 0, nffasa = 0;
    
    while ( (c = getopt(argc, argv, "n:g:d:m:x:")) != -1)
    {
        switch (c)
        {
        case 'n':
            pop_count = atoi(optarg);
            break;
        case 'g':
            max_gen = atoi(optarg);
            break;
        case 'd':
            dimension = atoi(optarg);
            break;
        case 'm':
            min = atof(optarg);
            break;
        case 'x':
            max = atof(optarg);
            break;
        case '?':
            printf("FFlies usage: -n NumberOfFlies -g NumberOfGenerations -d NumberOfDimension -m Min -x Max\n");
            return EXIT_FAILURE;
            break;
        default:
            printf("FFlies usage: -n NumberOfFlies -g NumberOfGenerations -d NumberOfDimension -m Min -x Max\n");
            break;
        }
    }

    mins = (double*)calloc(dimension, sizeof(double));
    maxs = (double*)calloc(dimension, sizeof(double));

    for (i=0; i < dimension; i++)
    {
        mins[i] = min;
        maxs[i] = max;
    }
    
    
    nffa   = test_ffa(pop_count, max_gen, dimension, mins, maxs, &akley);
    nffasa = test_ffasa(pop_count, max_gen, dimension, mins, maxs, &akley);

    printf("FFA: %ld, FFASA: %ld\n", nffa, nffasa);
    
    
    //ffa(pop_count, max_gen, dimension, mins, maxs, &rosenbrock);
    
    free(mins);
    free(maxs);
    return EXIT_SUCCESS;
};

double
yang(const ffly *fly, const size_t nparams)
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
akley(const ffly *fly, const size_t nparams)
{
    unsigned int i = 0;
    double exp1, exp2, frac;
    double sumsq = 0.0, sumcos = 0.0;
    
    for (i = 0; i < nparams; i++)
    {
        sumsq += fly->params[i] * fly->params[i];
        sumcos += cos(2 * PI * fly->params[i]);
    }
    frac = 1.0 / ((double) nparams);
    exp1 = exp(-0.2 * sqrt(frac * sumsq));
    exp2 = exp(frac * sumcos);
    
    return 1.0 / (-20 * exp1 - exp2 + 20 + E);
};
        
double
schwefel(const ffly *fly, const size_t nparams)
{
    unsigned int i = 0;
    const double a = 418.9829;
    double sum = 0.0;
    
    for (i = 0; i < nparams; i++)
    {
        sum +=  fly->params[i] * sin(sqrt(abs(fly->params[i])));
    }
    
    return 1.0 / (a * ((double)nparams) * sum);
};
        
double
rosenbrock(const ffly *fly, const size_t nparams)
{
    unsigned int i = 0;
    double sum = 0.0;
    double part = 0.0;
    double x;
    for (i=0; i < nparams-1; i++)
    {
        x = fly->params[i];
        
        part = (fly->params[i+1] - (x * x));
        part *= part;
        sum += (100 * part) + ( (x - 1) * (x - 1) );
    }
    return 1.0 / sum;
};
        
