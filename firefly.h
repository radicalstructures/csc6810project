#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <time.h>

/* 
    Some useful Macros
*/
#define POP_COUNT   50
#define MAX_GEN     50

typedef struct _ff
{
	double *params;
} ffly;

/* 
    Our type declarations
*/
typedef struct _ffp
{
	size_t nfflies;
	size_t nparams;
	ffly *flies;
} ffly_population;

/* 
    This will allow for different functions to be passed in for evaluation
*/
typedef double (obj_func)(const ffly*, const size_t nparams);

/* 
    This will initiate a population of fireflies
*/
ffly_population* 
init_fflies(const size_t ncount, const size_t nparams, const double mins[], const double maxs[]);

/*
    Use this to clean up your population
*/
void 
destroy_fflies(ffly_population *pop);

/*
    Our main function
*/
void 
ffa(const size_t nfireflies, const size_t niteration, const size_t nparams, const double mins[], const double maxs[], obj_func f);

