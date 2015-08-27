#ifndef INIT_H
#define INIT_H

#include <hdf5.h>
#include <hdf5_hl.h>

#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <random>

#include <cmath>

#ifndef SOL
#define SOL 299792458.0f
#endif
#ifndef ELEMENTARY_CHARGE
#define ELEMENTARY_CHARGE (1.602176565e-19)
#endif

typedef struct part{

  long double *x, *px, *q, *m;


} particle;

void init(	long double* t_start, long double *t_end, long double *dt,
            long double *beamspeed, long double *circumference,
            int *length,
            particle *p,
			long double *freq
) {
    //loop-variable for later use
    int i;

    //initialise length (from array length): number of particles
    *length = 1e0;
    
    *t_start = 0;//in seconds
    *t_end   = 1e-6;//in seconds
    *dt      = 1e-9;//in seconds
    
    *beamspeed = 0.467 * SOL;
    *circumference = 108.5;//m

	const long double amplitude = 5;//unit: meter
    *freq = 1e2;//unit: hearts
    const long double omega = 2 * M_PI * (*freq);
    const long double deltaOmega = 2 * M_PI * 1e5 * 0;

    std::random_device generator;
    //get the first standard distribution: mean, standart deviation
    //as the momentum is expected in eV: both quantities also in eV
    /*
     * Gaussian: E_total = E_beam + E_bucket. E_bucket = 0 for the syn-
     * cronous particle. as the energie in the bucket here is distributed,
     * the mean is 0 and the standard deviation is equal to the deviation
     * of the energie, whereas delta E / E = beta**2 * delta p / p with
     * delta p / p = 1e-5. E = 122 MeV/u, 12C3+ are considered
     */
    std::normal_distribution<long double> position(0, 0);
    std::normal_distribution<long double> circularFrequency(omega, deltaOmega);
    
    //allocate memory for each component of position
    p->x = (long double*) malloc(sizeof(long double) * (*length));

    //allocate memory for each component of momentum
    p->px = (long double*) malloc(sizeof(long double) * (*length));

    //allocate memory for mass and charge
    p->q = (long double*) malloc(sizeof(long double) * (*length));
    p->m = (long double*) malloc(sizeof(long double) * (*length));

    //initialise each parameter for each particle
    for(i=0; i < (*length); i++) {
		p->x[i]  = position(generator);//in m
        p->q[i] = -1;//in number of the elementary charge
        p->m[i] = 0.5e6;//in eV
        p->px[i] = 5.2;
        
    }
    
    printf("paricel 1: x = %Le\n", p->x[0]);
    printf("paricel 1: p = %Le\n", p->px[0]);
    
    double *tmp;
    tmp = (double*) malloc(sizeof(double) * (*length));
    for(i=0; i < (*length); i++) { tmp[i] = (double) p->px[i]; }
    
    hid_t file_id;
	hsize_t dims[1];
    file_id = H5Fcreate("distribution.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    dims[0] = (*length);
	H5LTmake_dataset(file_id,"/signal",1,dims,H5T_NATIVE_DOUBLE,tmp);
    dims[0] = 2;
    tmp[0] = (*dt);  tmp[1] = (double) *freq; 
    H5LTmake_dataset(file_id,"/params",1,dims,H5T_NATIVE_DOUBLE,tmp);
    H5Fclose(file_id);
	free(tmp);
}


#endif
