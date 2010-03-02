#include "firefly.h"

static void move_fflies(ffly_population *pop, const ffly_population *pop_old, obj_func f, 
                        const double alpha, const double gamma, const double mins[], const double maxs[]);
static void move_fly(ffly *fly, ffly *old, obj_func f, const size_t nparams, 
        const double alpha, const double gamma, const double mins[], const double maxs[]);
static void memcpy_fflies(ffly_population *fflies_old, ffly_population *dest);
static void memcpy_ffly(ffly *fly, ffly *dest, size_t nparams);
static void output_points(ffly_population *pop, const char *fname);
static double calc_distance(const ffly *fly, const ffly *fly_old, size_t nparams);
static double my_rand(void);
/*

        This creates our firefly population and assigns random positions.
*/
ffly_population*
init_fflies(const size_t ncount, const size_t nparams, const double mins[], const double maxs[])
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

void
ffa(const size_t nfireflies, const size_t niteration, const size_t nparams, const double mins[], const double maxs[],
    obj_func f)
{

    register unsigned int i = 0;
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

    output_points(fflies, "start.dat");
    for (i=0; i < niteration; i++)
    {
        //keep another copy for move function
        memcpy_fflies(fflies_old, fflies);

        //move the flies based on attractiveness
        move_fflies(fflies, fflies_old, f, alpha, gamma, mins, maxs);
    }
    output_points(fflies, "end.dat");

    destroy_fflies(fflies);
    destroy_fflies(fflies_old);
    return;    
};

/*

    Use this to copy a set of fireflies to a new set
*/
static void
memcpy_fflies(ffly_population *dest, ffly_population *fflies_old)
{
    register int i = 0;

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

        This is what moves our fireflies towards the most attractive flies
*/
static void
move_fflies(ffly_population *pop, const ffly_population *pop_old, obj_func f,
            const double alpha, const double gamma, const double mins[], const double maxs[])
{
    unsigned register int i = 0, j = 0;

    size_t nflies = pop->nfflies;
    size_t nparams = pop->nparams;

    for (i=0; i < nflies; i++)
    {
	    #pragma omp parallel for private(j) shared(nflies, pop, pop_old, f, nparams, alpha, gamma, i, mins, maxs)
        for (j = 0; j < nflies; j++)
        {
            move_fly(&pop->flies[i], &pop_old->flies[j], f, nparams, alpha, gamma, mins, maxs);
        }
    }
    return;
};

static void
move_fly(ffly *fly, ffly *old, obj_func f, const size_t nparams, 
        const double alpha, const double gamma, const double mins[], const double maxs[])
{
    unsigned register int i = 0;
    const double beta0 = 1.0;
    
    double ilight = (*f)(fly, nparams);
    double jlight = (*f)(old, nparams);

    if (ilight < jlight)
    {
        //get the distance to the other fly
        double r = calc_distance(fly, old, nparams);

        //determine attractiveness with air density [gamma]
        double beta = beta0 * exp((-gamma) * (r * r));

        //adjust position with a small random step
        for (i = 0; i < nparams; i++)
        {
            double val = ((1 - beta) * fly->params[i]) + (beta * old->params[i]) + (alpha * (my_rand() - .5));
            
            //keep within bounds
            fly->params[i] = (val < mins[i]) ? mins[i] : (val > maxs[i]) ? maxs[i] : val;
        }
    }
};

/*
    Calculates the euclidean distance of the two vectors
*/
static double
calc_distance(const ffly *fly, const ffly *fly_old, const size_t nparams)
{
    register unsigned int i = 0;
    double aggr = 0.0, dist = 0.0;
    for (i = 0; i < nparams; i++)
    {
        dist = fly->params[i] - fly_old->params[i];
        aggr += (dist * dist);
    }
    return sqrt(aggr);
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

double
my_rand(void)
{
    return ( ( (double)rand() ) / ((double) RAND_MAX + 1.0 ));
};

