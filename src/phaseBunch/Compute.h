#ifndef COMPUTE_H
#define COMPUTE_H

#include <math.h>
#include <stdio.h>

#include <assert.h>

#include "Prints.h"


#ifndef SOL
#define SOL 299792458
#endif

void updateParticle(
    long double dt,
    long double *x, long double *px,
    long double q, long double m
) {

    long double gamma,F;

    gamma = sqrtl(1 + ((*px)*(*px)) / (m*m));
    F = q * 2.98e26* (*x); //computes Lorentz-force

	//update position and momentum
    *x = *x + SOL * (*px) / ( gamma * m ) * dt + 1.0 / 2.0 * dt*dt * F * SOL * SOL / ( gamma * m );
    printf("%Le \n",*x);
    *px = *px + 3e8 * F * dt * gamma;
    
}

void compute(
    long double t_start, long double t_end, long double dt,
    long double x[], long double px[], 
    long double m[], long double q[],
    int len,
    long double beamspeed, long double circumference,
    hid_t* dataset
) {
    int i,j;
    long double t;
    for( t = t_start,j = 1; t < t_end - dt; t += dt) {

#pragma omp parallel for default(none) private(i) shared(len, x, px, dt, m, q) if(len > 4)
        for(i = 0; i < len; i++) {
			updateParticle(dt, &(x[i]), &(px[i]), q[i], m[i]);
        }
        
        //check, whether sync-particle passed the detector
        if( t * beamspeed > j * circumference ) {
			//store the current particle-positions, corrected by the current time
			//(the particle's offset to the sync-particles are computed)
		    SaveChunk( dataset, j, len, beamspeed, &x);
			j++;
		}

    }

	printf("j = %i\n", j);
}
#endif
