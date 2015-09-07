#ifndef INIT_H
#define INIT_H

#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include <assert.h>

#include "Prints.h"

#define CUDA_RETURN_CHECK(A) if(A != cudaSuccess) { printf("CUDA-Error at file %s, line %i\n", __FILE__, __LINE__); exit(1); }

typedef struct part{
   double *x, *px, *q, *m;
   double *dev_x, *dev_px, *dev_q, *dev_m;
   double *dev_time;
} particle;

double ranf() {
        return (double) rand() / RAND_MAX;
}

/* boxmuller.c           Implements the Polar form of the Box-Muller
                         Transformation

                      (c) Copyright 1994, Everett F. Carter Jr.
                          Permission is granted by the author to use
			  this software for any application provided this
			  copyright notice is preserved.

*/

double box_muller(double m, double s)      /* normal random variate generator */
{                                       /* mean m, standard deviation s */
        double x1, x2, w, y1;
        static double y2;
#warning "Use of static variable - NOT threadsafe"
        static int use_last = 0;

        if (use_last)                   /* use value from previous call */
        {
                y1 = y2;
                use_last = 0;
        }
        else
        {
                do {
                        x1 = 2.0 * ranf() - 1.0;
                        x2 = 2.0 * ranf() - 1.0;
                        w = x1 * x1 + x2 * x2;
                } while ( w >= 1.0 );

                w = sqrt( (-2.0 * log( w ) ) / w );
                y1 = x1 * w;
                y2 = x2 * w;
                use_last = 1;
        }

        return( m + y1 * s );
}


void init(	int length, particle *p, double deltaP,
			hid_t *file, hid_t *dataset) {
    
    //allocate memory for position, momentum, mass & charge
    p->x  = (double*) malloc(sizeof(double) * length);
    p->px = (double*) malloc(sizeof(double) * length);
    p->q  = (double*) malloc(sizeof(double) * length);
    p->m  = (double*) malloc(sizeof(double) * length);
    
	//mem > 3GB should be chunked -> not implemented
	printf("GPU-Mem to be allocated: % .2lf\n", (double) length * 4 * sizeof(double));
	assert( (double) length * 4 * sizeof(double) < 1024.f * 1024.f * 1024.f * 3.f);
	
	CUDA_RETURN_CHECK( cudaMalloc( (void**) &(p->dev_x), length * sizeof(double)));
	CUDA_RETURN_CHECK( cudaMalloc( (void**) &(p->dev_px), length * sizeof(double)));
	CUDA_RETURN_CHECK( cudaMalloc( (void**) &(p->dev_m), length * sizeof(double)));
	CUDA_RETURN_CHECK( cudaMalloc( (void**) &(p->dev_q), length * sizeof(double)));
	CUDA_RETURN_CHECK( cudaMalloc( (void**) &(p->dev_time), length * sizeof(double)));

    //initialise each parameter for each particle
    for( int i=0; i < length; i++) {
		p->x[i]  = 0;//in m
        p->q[i] = -1;//in number of the elementary charge
        p->m[i] = 0.5e6;//in eV
        p->px[i] = box_muller(0, deltaP);
        
    }
    
    CUDA_RETURN_CHECK( cudaMemcpy( p->dev_x, 	p->x, 	length * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_RETURN_CHECK( cudaMemcpy( p->dev_px, 	p->px, 	length * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_RETURN_CHECK( cudaMemcpy( p->dev_m, 	p->m, 	length * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_RETURN_CHECK( cudaMemcpy( p->dev_q, 	p->q, 	length * sizeof(double), cudaMemcpyHostToDevice));
    
    printInitDistribution(&(p->px), length);
    InitResultFile(file, dataset, length);
  
}


#endif
