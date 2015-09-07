#include <stdio.h>

#include "Init.h"
#include "Destruct.h"
#include "Compute.h"


#ifndef SOL
#define SOL 299792458.0f
#endif

int main() {
    printf("Initialising...\n");
    
    //start time, end time and time resolution (each in seconds)
    long double t_start = 0,
				t_end	= 1e-2,
				dt		= 1e-7;
    long double beamspeed 		= 0.47 * SOL,
				circumference	= 108.5, //m
				deltaP = 146.0;//eV
    int 		len = 1e4; //number of particles
    particle p;
    
    //HDF5-Handles;
    hid_t   file;
    hid_t   dataset;

    init( 	len, &p, deltaP,
			&file, &dataset);

    printf("Computing...\n");
    compute(t_start, t_end, dt,
        p.x, p.px,
        p.m,p.q, len,
        beamspeed, circumference,
        &dataset);
        
    printf("Clean up...\n");
    FinalizeResultFile(&dataset, &file,
		dt, circumference, beamspeed);
    destruct(p);

    return EXIT_SUCCESS;
}
