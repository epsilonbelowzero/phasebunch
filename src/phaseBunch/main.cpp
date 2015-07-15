#include <stdio.h>
//#include <omp.h>

#include "Init.h"
#include "Destruct.h"
#include "Compute.h"
#include "Params.h"

int main(int argc, char** argv) {

//    printf("Using %i Threads\n", omp_get_num_procs());
    printf("Initialising...\n");

    long double t_start, t_end, dt;
    long double beamspeed, circumference;
    long double freq;
    int len, k;
    particle p;


    init(&t_start, &t_end, &dt,
        &beamspeed, &circumference,
        &len,
        &p,
		&freq
    );

    printf("Computing...\n");
    compute(t_start, t_end, dt,
        p.x, p.px,
        p.m,p.q, len,
        &k,
        beamspeed, circumference,
	&freq);

    destruct(p);

    return EXIT_SUCCESS;
}
