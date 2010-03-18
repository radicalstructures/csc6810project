#include "firefly.h"

/* 
    Static functions not to be used outside of this file 
*/
static void move_fflies(ffly_population *pop, const ffly_population *pop_old, obj_func f, 
                        double alpha, const double gamma, const double mins[], const double maxs[]);
static void move_fly(ffly *fly, ffly *old, obj_func f, const size_t nparams, 
                        double alpha, const double gamma, const double mins[], const double maxs[]);

static ffly_population* init_fflies(const size_t ncount, const size_t nparams, const double mins[], 
                        const double maxs[]);
                        
static void destroy_fflies(ffly_population *pop);
static void memcpy_fflies(ffly_population *fflies_old, ffly_population *dest);
static void memcpy_ffly(ffly *fly, ffly *dest, size_t nparams);
static void output_points(ffly_population *pop, const char *fname);

static double calc_distance(const ffly *fly, const ffly *fly_old, size_t nparams);
static double my_rand(void);
static double std_dev(const ffly_population *pop);
static double mean_delta(const ffly_population *pop, const ffly_population *old);

/*
        This creates our firefly population and assigns random positions.
*/
static ffly_population*
init_fflies(const size_t ncount, const size_t nparams, const double mins[], const double maxs[])
{
    size_t i = 0, j = 0;
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
            pop->flies[i].val = 0.0;
        }
    }

    return pop;
};

/*

        This is called to free up the population
*/
static void
destroy_fflies(ffly_population *pop)
{
    size_t i = 0;
    for (i = 0; i < pop->nfflies; i++)
    {
        free(pop->flies[i].params);
    }
    free(pop->flies);
    free(pop);

    return;
};

/*

    Use this to copy a set of fireflies to a new set
*/
static void
memcpy_fflies(ffly_population *dest, ffly_population *fflies_old)
{
    size_t i = 0;

    dest->nfflies = fflies_old->nfflies;
    dest->nparams = fflies_old->nparams;

    for (i = 0 ; i < fflies_old->nfflies; i++)
    {
        dest->flies[i].val = fflies_old->flies[i].val;
        memcpy_ffly(&dest->flies[i], &fflies_old->flies[i], dest->nparams);
    }
};

static void
memcpy_ffly(ffly *dest, ffly *fly, size_t nparams)
{
    memcpy(dest->params, fly->params, sizeof(double) * nparams);
};

void
ffa(const size_t nfireflies, const size_t niteration, const size_t nparams, const double mins[], const double maxs[],
    obj_func f)
{

    size_t i = 0;
    ffly_population *fflies = NULL;
    ffly_population *fflies_old = NULL;

    const double alpha = ALPHA_ZERO;   //randomness step
    const double gamma = GAMMA;         //absorption coefficient

    //initialize our RNG
    srand(time(NULL));

    //initialize our firefly array
    fflies = init_fflies(nfireflies, nparams, mins, maxs);

    //initialize our old firefly array
    fflies_old = init_fflies(nfireflies, nparams, mins, maxs);

    output_points(fflies, "start_ffa.dat");
    for (i=0; i < niteration; i++)
    {
        //keep another copy for move function
        memcpy_fflies(fflies_old, fflies);

        //move the flies based on attractiveness
        move_fflies(fflies, fflies_old, f, alpha, gamma, mins, maxs);
    }
    output_points(fflies, "end_ffa.dat");

    destroy_fflies(fflies);
    destroy_fflies(fflies_old);
    return;    
};

size_t
test_ffa(const size_t nfireflies, const size_t nparams, const double mins[], const double maxs[],
    obj_func f)
{
    size_t i = 0;
    ffly_population *fflies = NULL;
    ffly_population *fflies_old = NULL;

    double min = 0.0, min_old = 0.0;
    const double alpha = ALPHA_ZERO;    //randomness step
    const double gamma = GAMMA;         //absorption coefficient

    //initialize our RNG
    srand(time(NULL));

    //initialize our firefly array
    fflies = init_fflies(nfireflies, nparams, mins, maxs);

    //initialize our old firefly array
    fflies_old = init_fflies(nfireflies, nparams, mins, maxs);

    output_points(fflies, "start_ffa.dat");
    do
    {
        min_old = min;
        //keep another copy for move function
        memcpy_fflies(fflies_old, fflies);

        //move the flies based on attractiveness
		move_fflies(fflies, fflies_old, f, alpha, gamma, mins, maxs);
        i++;
    } while (mean_delta(fflies, fflies_old) > EPSILON);
    output_points(fflies, "end_ffa.dat");

    destroy_fflies(fflies);
    destroy_fflies(fflies_old);

    return i * nfireflies;    
};

void
ffasa(const size_t nfireflies, const size_t niteration, const size_t nparams, const double mins[], const double maxs[],
    obj_func f)
{

    size_t i = 0;
    ffly_population *fflies = NULL;
    ffly_population *fflies_old = NULL;

    double alpha = 0.0;
    const double alpha0 = ALPHA_ZERO;   //intial randomness step
    const double gamma = GAMMA;         //absorption coefficient

    //initialize our RNG
    srand(time(NULL));

    //initialize our firefly array
    fflies = init_fflies(nfireflies, nparams, mins, maxs);

    //initialize our old firefly array
    fflies_old = init_fflies(nfireflies, nparams, mins, maxs);

    output_points(fflies, "start_ffasa.dat");
    for (i=0; i < niteration; i++)
    {
        //keep another copy for move function
        memcpy_fflies(fflies_old, fflies);

        //calculate our new alpha
        alpha = alpha0 / log(i + 1);
        
        //move the flies based on attractiveness
        move_fflies(fflies, fflies_old, f, alpha, gamma, mins, maxs);
    }
    output_points(fflies, "end_ffasa.dat");

    destroy_fflies(fflies);
    destroy_fflies(fflies_old);
    return;    
};

size_t
test_ffasa(const size_t nfireflies, const size_t nparams, const double mins[], const double maxs[],
    obj_func f)
{

    size_t i = 1;
    ffly_population *fflies = NULL;
    ffly_population *fflies_old = NULL;

    double min = 0.0, min_old = 0.0;
    double alpha = 0.0;
    const double alpha0 = ALPHA_ZERO;   //intial randomness step
    const double gamma = GAMMA;         //absorption coefficient

    //initialize our RNG
    srand(time(NULL));

    //initialize our firefly array
    fflies = init_fflies(nfireflies, nparams, mins, maxs);

    //initialize our old firefly array
    fflies_old = init_fflies(nfireflies, nparams, mins, maxs);

    output_points(fflies, "start_ffasa.dat");
    do
    {
        min_old = min;
        //keep another copy for move function
        memcpy_fflies(fflies_old, fflies);

        //calculate our new alpha
        alpha = alpha0 / log(i + 1);

        //move the flies based on attractiveness
        move_fflies(fflies, fflies_old, f, alpha, gamma, mins, maxs);
        i++;
    } while (mean_delta(fflies, fflies_old) > EPSILON);
    
    output_points(fflies, "end_ffasa.dat");

    destroy_fflies(fflies);
    destroy_fflies(fflies_old);
    
    return (i-1) * nfireflies;    
};


/*
        This is what moves our fireflies towards the most attractive flies. Vanilla FFA
*/
static void
move_fflies(ffly_population *pop, const ffly_population *pop_old, obj_func f,
            double alpha, const double gamma, const double mins[], const double maxs[])
{
    size_t i = 0, j = 0;

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
};

/*
    This moves an individual fly torwards another
*/
static void
move_fly(ffly *fly, ffly *old, obj_func f, const size_t nparams, 
        const double alpha, const double gamma, const double mins[], const double maxs[])
{
    size_t i = 0;
    const double beta0 = BETA_ZERO;	//base attraction
    
    fly->val = (*f)(fly, nparams);
    double jlight = (*f)(old, nparams);

    if (fly->val > jlight)
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
    size_t i = 0;
    double aggr = 0.0, dist = 0.0;
    for (i = 0; i < nparams; i++)
    {
        dist = fly->params[i] - fly_old->params[i];
        aggr += (dist * dist);
    }
    return sqrt(aggr);
};

/*
    writes our points out to a file
*/
static void 
output_points(ffly_population *pop, const char *fname)
{

    size_t i = 0, j = 0;
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

/*
    returns value in [0, 1]
*/
static double
my_rand(void)
{
    return ( ( (double)rand() ) / ((double) RAND_MAX ));
};

/*
    returns the std dev of the objective func values
*/
static double 
std_dev(const ffly_population *pop)
{
    size_t i = 0;
    double mean = 0.0;
    double sumsq = 0.0;
    
    for (i = 0; i < pop->nfflies; i++)
    {
        mean += pop->flies[i].val;
    }
    
    mean /= ((double)pop->nfflies);
    
    for (i = 0; i < pop->nfflies; i++)
    {
        sumsq += (pop->flies[i].val - mean) * (pop->flies[i].val - mean);
    }
    
    sumsq /= ((double)pop->nfflies);
    
    return sqrt(sumsq);
};

/*
    returns the mean of all the deltas for the objective func
*/
static double
mean_delta(const ffly_population *pop, const ffly_population *old)
{
    double sumdelta = 0.0;
    size_t i = 0;
    
    for (i = 0; i < pop->nfflies; i++)
    {
        sumdelta += fabs(old->flies[i].val - pop->flies[i].val);
    }
    
    return sumdelta / ((double)pop->nfflies);
};

