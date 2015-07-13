#ifndef COMPUTE_H
#define COMPUTE_H

//~ #include <math.h>
#include <cmath>
#include <stdio.h>

//#include <omp.h>

#include "Params.h"
#include "Prints.h"
#include "Signal.h"

#ifndef SOL
#define SOL 299792458
#endif

long double computeGamma(long double px, long double m) {
    return std::sqrt(1 - (px*px) / (m*m));
}

long double computeVi(long double pi, long double gamma, long double m) {
    //v_i = p_i * c / (gamma * m)
    return SOL * pi / (gamma * m);
}


void computeLorentz( long double q, long double x, long double *F, long double t) {
    *F = abs(q) *  Ex(x, t);
}

void computeNewImpulse( long double dt, long double *px, long double F ) {
    //old computation of new momentum: no energy conservation (magnetic field also accelerates)
    *px = *px + 3e8 * F * dt;
}

long double computeNewPosition(
    long double dt, long double *x, long double px, long double F, long double gamma, long double m
) {
    *x += SOL * px / ( gamma * m ) * dt + 1.0 / 2.0 * dt*dt * F * SOL * SOL / ( gamma * m );
}

void updateParticle(
    long double t, long double dt,
    long double *x, long double *px,
    long double q, long double m
) {

    long double gamma,vx,F;

    gamma = computeGamma(*px, m);
    computeLorentz(q, *x, &F, t);

    computeNewPosition(dt, x, *px, F, gamma, m);
    computeNewImpulse(dt, px, F);
    
}

void compute(
    long double t_start, long double t_end, long double dt,
    long double x[], long double px[], 
    long double m[], long double q[],
    int len, int *k, long double **times,
    long double beamspeed, long double circumference,
    long double *freq
) {

    int i,j, l;

    long double t, I;
    long double h = 1 / ((*freq) * dt);
    for( t = t_start,j = 0; t < t_end - dt; t += dt, j++) {

#pragma omp parallel for default(none) private(i) shared(len, x, px, dt, m, q, t)
        for(i = 0; i < len; i++) {
			updateParticle(t, dt, &(x[i]), &(px[i]), q[i], m[i]);
        }

    }

    *k = j;
}
#endif
