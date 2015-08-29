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
    double t_start = 0,
				t_end	= 1e-5,
				dt		= 1e-9;
    double beamspeed 		= 0.467 * SOL,
				circumference	= 108.5, //m
				deltaP = 26e3;
    int 		len = 1e4; //number of particles
    particle p;
    
    //HDF5-Handles;
    hid_t   file;
    hid_t   dataset;

    init( 	len, &p, deltaP,
			&file, &dataset);

    printf("Computing...\n");
    compute(t_start, t_end, dt,
        &p, len,
        beamspeed, circumference,
        &dataset);
        
    printf("Clean up...\n");
    FinalizeResultFile(&dataset, &file,
		dt, circumference, beamspeed);
    destruct(p);

    return EXIT_SUCCESS;
}
