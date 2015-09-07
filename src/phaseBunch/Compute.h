#ifndef COMPUTE_H
#define COMPUTE_H

#include <math.h>
#include <stdio.h>

#include <assert.h>

#include "Prints.h"
#include "Init.h"


#ifndef SOL
#define SOL 299792458
#endif

__global__ void updateParticleCUDA(double dt, double* x, double* px, double* q, double* m, int length) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < length) {
		double gamma,F;

		gamma = sqrt(1 + (px[tid] * px[tid]) / (m[tid] * m[tid]));
		F = q[tid] * 2.193245422464302e-06* x[tid]; //computes Lorentz-force

		//update position and momentum
		x[tid] = x[tid] + SOL * px[tid] / ( gamma * m[tid] ) * dt + 1.0 / 2.0 * dt*dt * F * SOL * SOL / ( gamma * m[tid] );
		px[tid] = px[tid] + 3e8 * F * dt * gamma;
		
		tid += gridDim.x * blockDim.x;
	}
}

__global__ void computeTime(double* x, double* t, double beamspeed, int length) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while(tid < length) {
        t[tid] = x[tid] / beamspeed
    }
}

void compute(
    long double t_start, long double t_end, long double dt,
    particle* p,
    int len,
    long double beamspeed, long double circumference,
    hid_t* dataset
) {
    int j;
    long double t;
    
    for( t = t_start,j = 1; t < t_end - dt; t += dt) {
        
        updateParticleCUDA<<<128, 256>>>(dt, p->dev_x, p->dev_px, p->dev_q, p->dev_m, len);
        
        //check, whether sync-particle passed the detector
        if( t * beamspeed > j * circumference ) {
			//store the current particle-positions, corrected by the current time
			//(the particle's offset to the sync-particles are computed)
			
            computeTime<<<128, 256>>>(p->dev_x, p->dev_time, (double) beamspeed, length);
			cudaMemcpy( p->x, p->dev_time, sizeof(double) * len, cudaMemcpyDeviceToHost);
			
		    SaveChunk( dataset, j, len, beamspeed, &(p->x));
			j++;
		}

    }

	printf("j = %i\n", j);
}
#endif
