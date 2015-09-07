#ifndef INIT_H
#define INIT_H

#include <stdlib.h>
#include <stdio.h>
#include <random>

#include <math.h>

#include "Prints.h"

typedef struct part{
  long double *x, *px, *q, *m;
} particle;

void init(	int length, particle *p, long double deltaP,
			hid_t *file, hid_t *dataset) {

    std::random_device generator;
    //get the first standard distribution: mean, standart deviation
    //as the momentum is expected in eV: both quantities also in eV
    std::normal_distribution<long double> position(0, 0);
    std::normal_distribution<long double> momentum(0, deltaP);
    
    //allocate memory for position, momentum, mass & charge
    p->x  = (long double*) malloc(sizeof(long double) * length);
    p->px = (long double*) malloc(sizeof(long double) * length);
    p->q  = (long double*) malloc(sizeof(long double) * length);
    p->m  = (long double*) malloc(sizeof(long double) * length);

    //initialise each parameter for each particle
    for( int i=0; i < length; i++) {
		p->x[i]  = position(generator);//in m
        p->q[i] = 6;//in number of the elementary charge
        p->m[i] = 11.26659e9;//in eV
        p->px[i] = momentum(generator);
        
    }
    
    printInitDistribution(&(p->px), length);
    InitResultFile(file, dataset, length);
  
}


#endif
