#include "firefly.h"

static void move_fflies(ffly_population *pop, const ffly_population *pop_old, obj_func f, double alpha, double gamma, double mins[], double maxs[]);
static void move_fly(ffly *fly, ffly *old, obj_func f, double distances[], size_t nparams, double alpha, double gamma);
static void memcpy_fflies(ffly_population *fflies_old, ffly_population *dest);
static void memcpy_ffly(ffly *fly, ffly *dest, size_t nparams);
static double calc_distance(double *dest, size_t nparams, const ffly *fly, const ffly *fly_old);
static void fix_positions(ffly_population *pop, double mins[], double maxs[]);
static void output_points(ffly_population *pop, const char *fname);
static void print_fflies(ffly_population *pop);

/*

        This creates our firefly population and assigns random positions.
*/
ffly_population*
init_fflies(size_t ncount, size_t nparams, double mins[], double maxs[])
{
    register unsigned int i = 0, j = 0;
    ffly_population *pop = NULL;

    //create memory chunks for values
    pop = (ffly_population*)malloc(sizeof(ffly_population));
    pop->nfflies = ncount;
    pop->nparams = nparams;
    pop->flies = (ffly*)calloc(ncount, sizeof(ffly));

    //init random positions and zero out light value
    for (i=0; i < ncount; i++)
    {
        pop->flies[i].params = (double*)calloc(nparams, sizeof(double));
        for (j=0; j < nparams; j++)
        {
            pop->flies[i].params[j] = my_rand()*(maxs[j]-mins[j]) + mins[j];
        }
    }

    return pop;
};

/*

        This is called to free up the population
*/
void
destroy_fflies(ffly_population *pop)
{
    register unsigned int i = 0;
    for (i = 0; i < pop->nfflies; i++)
    {
        free(pop->flies[i].params);
    }
    free(pop->flies);
    free(pop);

    return;
};

point
ffa(size_t nfireflies, size_t niteration, size_t nparams, double mins[], double maxs[],
    obj_func f)
{

    register unsigned int i = 0;
    point p;
    size_t size = 0;
    ffly_population *fflies = NULL;
    ffly_population *fflies_old = NULL;

    const double alpha = 0.2; //randomness step
    const double gamma = 1.0; //absorption coefficient

    //initialize our RNG
    srand(time(NULL));

    //initialize our firefly array
    fflies = init_fflies(nfireflies, nparams, mins, maxs);

    //initialize our old firefly array
    fflies_old = init_fflies(nfireflies, nparams, mins, maxs);

    size = sizeof(double) * nfireflies;

    output_points(fflies, "start.dat");
    for (i=0; i < niteration; i++)
    {
        //keep another copy for move function
        memcpy_fflies(fflies, fflies_old);

        //move the flies based on attractiveness
        move_fflies(fflies, fflies_old, f, alpha, gamma, mins, maxs);
    }
    output_points(fflies, "end.dat");

    destroy_fflies(fflies);
    destroy_fflies(fflies_old);
    
    return p;
};

/*

    Use this to copy a set of fireflies to a new set
*/
static void
memcpy_fflies(ffly_population *dest, ffly_population *fflies_old)
{
    register int i = 0;

    memcpy(dest->flies, fflies_old->flies, sizeof(ffly)*(fflies_old->nfflies));
    dest->nfflies = fflies_old->nfflies;
    dest->nparams = fflies_old->nparams;

    for (i = 0 ; i < fflies_old->nfflies; i++)
    {
        memcpy_ffly(&dest->flies[i], &fflies_old->flies[i], dest->nparams);
    }
};

static void
memcpy_ffly(ffly *dest, ffly *fly, size_t nparams)
{
    memcpy(dest->params, fly->params, sizeof(double) * nparams);
};

/*

    Use this to just print values of the population
*/
void
print_fflies(ffly_population *pop)
{
    unsigned register int i = 0, j = 0;

    for (i = 0; i < pop->nfflies; i++)
    {
        printf("FFly %d:", i);
        for (j = 0; j < pop->nparams; j++)
        {
            printf("\t%.2lf", pop->flies[i].params[j]);
        }
        printf("\n");
    }
    return;
};

/*

        This is what moves our fireflies towards the most attractive flies
*/
static void
move_fflies(ffly_population *pop, const ffly_population *pop_old, obj_func f,
            double alpha, double gamma, double *mins, double *maxs)
{
    unsigned register int i = 0, j = 0;

    size_t nflies = pop->nfflies;
    size_t nparams = pop->nparams;
    double *distances = (double*)calloc(nparams, sizeof(double));

    for (i=0; i < nflies; i++)
    {
#pragma parallel for private(i)
        for (j = 0; j < nflies; j++)
        {
            if (j != i)
            {
                move_fly(&pop->flies[i], &pop_old->flies[j], f, distances, pop->nparams, alpha, gamma);
            }
        }
    }

    //fix boundaries overstepped by random step
    fix_positions(pop, mins, maxs);
    free(distances);
    return;
};

static void
move_fly(ffly *fly, ffly *old, obj_func f, double distances[], size_t nparams, double alpha, double gamma)
{
    unsigned register int i = 0;
    const double beta0 = 1.0;
    double ilight = (*f)(fly);
    double jlight = (*f)(old);

    if (ilight < jlight)
    {
        //get the distance to the other fly
        double r = calc_distance(distances, nparams, fly, old);

        //determine attractiveness with air density [gamma]
        double beta = beta0 * exp((-gamma) * (r * r));

        //adjust position with a small random step
        for (i = 0; i < nparams; i++)
        {
            fly->params[i] = ((1 - beta) * fly->params[i]) + (beta * old->params[i]) + (alpha * (my_rand() - .5));
        }
    }
};

static double
calc_distance(double *dest, size_t nparams, const ffly *fly, const ffly *fly_old)
{
    register unsigned int i = 0;
    double aggr = 0.0;
    for (i = 0; i < nparams; i++)
    {
        dest[i] = fly->params[i] - fly_old->params[i];
        aggr += (dest[i] * dest[i]);
    }
    return sqrt(aggr);
}

/*

        This should correct our positions that have past the boundries from random steps
 */
static void
fix_positions(ffly_population *pop, double *mins, double *maxs)
{
    unsigned register int i = 0, j = 0;

    for (i=0; i < pop->nfflies; i++)
    {
        for (j = 0; j < pop->nparams; j++)
        {
            if (pop->flies[i].params[j] < mins[j]) pop->flies[i].params[j] = mins[j];
            if (pop->flies[i].params[j] > maxs[j]) pop->flies[i].params[j] = maxs[j];
        }
    }
    return;
};

static void 
output_points(ffly_population *pop, const char *fname)
{

    unsigned register int i = 0, j = 0;
    FILE *file = NULL;

    file = fopen(fname, "wt");
    if (file != NULL)
    {
        for (i = 0; i < pop->nfflies; i++)
        {
            for (j = 0; j < pop->nparams; j++)
            {
                fprintf(file, "%.2lf ", pop->flies[i].params[j]);
            }
            fprintf(file, "\n");
        }
    }
    fclose(file);
};

