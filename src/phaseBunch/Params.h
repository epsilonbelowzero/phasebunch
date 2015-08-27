#ifndef PARAMS_H
#define PARAMS_H

//the factor is computed by sovling the harmonic oszillator
//and set a specific mass, charge and frequency, so
//k = omega² * m / q
//for 100 Hz, electron charge and mass k = 1.400967e13, [k] = V / (m^2)

inline long double Ex(long double x){
    
    return 1e-8*x;

}

#endif
