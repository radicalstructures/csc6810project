#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "point.h"

/* 
    Some useful Macros
*/
#define POP_COUNT   50
#define MAX_GEN     50

/* 
    Our type declarations
*/
typedef struct _ffp
{
	size_t nfflies;
	double *x_values;
	double *y_values;
	double *light_values;
} ffly_population;

/* 
    This will allow for different functions to be passed in for evaluation
*/
typedef void (obj_func)(ffly_population*);

/* 
    This will initiate a population of fireflies
*/
ffly_population* 
init_fflies(size_t ncount, point min, point max);

/*
    Use this to clean up your population
*/
void 
destroy_fflies(ffly_population *pop);

/*
    Our main function
*/
point 
ffa(size_t nfireflies, size_t niteration, point min, point max, obj_func f);

