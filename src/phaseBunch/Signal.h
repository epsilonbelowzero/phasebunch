#ifndef SIGNAL_H
#define SIGNAL_H

#ifndef SOL
#define SOL 3e8
#endif
#include <math.h>

bool check(long double t, int k, long double circumference, long double beamspeed) {

    if( t * beamspeed >= k * circumference) {
        return true;
    }

    return false;

}

void getSignals(
    long double t, int *k, int len,
    long double x[],
    long double beamspeed, long double circumference,
    long double times[]
)  {


        int i;
        *k = *k + 1;

	for(int i = 0; i < len;i++){

	if(x[i]>= 2*M_PI){

		
		times[i]=t;


	}

	

	}

}

#endif
