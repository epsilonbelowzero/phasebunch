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

long double computeGamma(long double px, long double py, long double pz, long double m) {
    return std::sqrt(1 - (px*px + py*py + pz*pz) / (m*m));
}

long double computeVi(long double pi, long double gamma, long double m) {
    //v_i = p_i * c / (gamma * m)
    return SOL * pi / (gamma * m);
}

void computeFb(
    long double vx, long double vy, long double vz,
    long double x, long double y, long double z,
    long double* Fx, long double *Fy, long double *Fz,
    long double t
) {
    //F_i,b = e_ikl * v_k * b_l
    *Fx = vy * Bz(x, y, z, t) - vz * By(x, y, z, t);
    *Fy = vz * Bx(x, y, z, t) - vx * Bz(x, y, z, t);
    *Fz = vx * By(x, y, z, t) - vy * Bx(x, y, z, t);
}

void computeLorentz(
    long double q,
    long double x, long double y, long double z,
    long double *Fx, long double *Fy, long double *Fz,
    long double t
) {
    *Fx = abs(q) *  Ex(x, y, z, t) + q * (*Fx);
    *Fy = abs(q) *  Ey(x, y, z, t) + q * (*Fy);
    *Fz = abs(q) *  Ez(x, y, z, t) + q * (*Fz);
}

void computeNewImpulse(
    long double dt,
    long double *px, long double *py, long double *pz,
    long double Fx, long double Fy, long double Fz
) {
    
    //energy-conservation!
    long double valueP = std::sqrt( (*px) * (*px) + (*py) * (*py) + (*pz) * (*pz));
    
    if( valueP == 0) {
        //no impulse -> there CANNOT be a magnetic force, so old computation:
        *px = *px + 3e8 * Fx * dt;
        *py = *py + 3e8 * Fy * dt;
        *pz = *pz + 3e8 * Fz * dt;
        
        return;
    }

    long double ePx = *px / valueP, //unit vector in impulse direction
                ePy = *py / valueP,
                ePz = *pz / valueP,

                fPx = (Fx * ePx + Fy * ePy + Fz * ePz) * ePx, // force parallel to impulse
                fPy = (Fx * ePx + Fy * ePy + Fz * ePz) * ePy,
                fPz = (Fx * ePx + Fy * ePy + Fz * ePz) * ePz,

                fSx = Fx - fPx, //force vertical to impulse
                fSy = Fy - fPy,
                fSz = Fz - fPz,


                px1 = *px + 3e8 * fPx * dt, //impulse part 1: parallel force part
                py1 = *py + 3e8 * fPy * dt,
                pz1 = *pz + 3e8 * fPz * dt,

                px2 = px1 + 3e8 * fSx * dt, //impulse part 2: whole impulse without energy conservation
                py2 = py1 + 3e8 * fSy * dt,
                pz2 = pz1 + 3e8 * fSz * dt;
/*
    //old computation of new momentum: no energy conservation (magnetic field also accelerates)
    *px = *px + 3e8 * Fx * dt;
    *py = *py + 3e8 * Fy * dt;
    *pz = *pz + 3e8 * Fz * dt;

*/

    *px = px2 / std::sqrt(px2*px2 + py2*py2 + pz2*pz2) * std::sqrt(px1*px1 + py1*py1 + pz1*pz1); //resulting impulse
    *py = py2 / std::sqrt(px2*px2 + py2*py2 + pz2*pz2) * std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
    *pz = pz2 / std::sqrt(px2*px2 + py2*py2 + pz2*pz2) * std::sqrt(px1*px1 + py1*py1 + pz1*pz1);

}

long double computeNewPosition(
    long double dt,
    long double *x, long double *y, long double *z,
    long double px, long double py, long double pz,
    long double Fx, long double Fy, long double Fz,
    long double gamma, long double m
) {
    *x += SOL * px / ( gamma * m ) * dt + 1.0 / 2.0 * dt*dt * Fx * SOL * SOL / ( gamma * m );
    *y += SOL * py / ( gamma * m ) * dt + 1.0 / 2.0 * dt*dt * Fy * SOL * SOL / ( gamma * m );
    *z += SOL * pz / ( gamma * m ) * dt + 1.0 / 2.0 * dt*dt * Fz * SOL * SOL / ( gamma * m );
}

void updateParticle(
    long double t, long double dt,
    long double *x, long double *y, long double *z,
    long double *px, long double *py, long double *pz,
    long double q, long double m
) {

    long double gamma,vx,vy,vz,Fx,Fy,Fz;

    gamma = computeGamma(*px, *py, *pz, m);
    vx = computeVi(*px, gamma, m);
    vy = computeVi(*py, gamma, m);
    vz = computeVi(*pz, gamma, m);

    computeFb(vx, vy, vz, *x, *y, *z, &Fx, &Fy, &Fz, t);
    computeLorentz(q, *x, *y, *z, &Fx, &Fy, &Fz, t);

    computeNewPosition(dt, x, y, z, *px, *py, *pz, Fx, Fy, Fz, gamma, m);
    computeNewImpulse(dt, px, py, pz, Fx, Fy, Fz);
    
}

void compute(
    long double t_start, long double t_end, long double dt,
    long double x[], long double y[], long double z[],
    long double px[], long double py[], long double pz[],
    long double m[], long double q[],
    int len, int *k, long double **times,
    long double beamspeed, long double circumference,
    long double *freq
) {

    int i,j, l;

    //~ double *tmp;
    //~ tmp = (double*) malloc(sizeof(double) * len);

    //hid_t file_id;
    //hsize_t dims[1];
    //double params[2] = { (double) dt, (double) *freq }; 
    //~ hid_t file_id;
	//~ hsize_t dims[1];
    //~ double params[2] = { (double) dt, (double) px[0] / 2 / M_PI};
    
    //~ char buf[20];

    long double t, I;
    long double h = 1 / ((*freq) * dt);
    for( t = t_start,j = 0; t < t_end - dt; t += dt, j++) {
        (*times)[2*j] = t;
        (*times)[2*j + 1] = 0;

#pragma omp parallel for default(none) private(i) shared(len, x, px, dt, times, j, freq, h)
        for(i = 0; i < len; i++) {
            x[i] += px[i] * dt;
            if( x[i] >= 2 * M_PI) {
                x[i] -= 2 * M_PI;
#pragma omp critical
                (*times)[2*j + 1] += h;
            }
        }    

      /* for(i=0; i < len; i++) { tmp[i] = (double) x[i]; }
       
        snprintf(buf, 20, "distribution%i.h5", j);
        file_id = H5Fcreate(buf,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
        dims[0] = len;
        H5LTmake_dataset(file_id,"/signal",1,dims,H5T_NATIVE_DOUBLE,tmp);
        dims[0] = 2;
        H5LTmake_dataset(file_id,"/params",1,dims,H5T_NATIVE_DOUBLE,params);
        H5Fclose(file_id);*/
        //~ for(i=0; i < len; i++) { tmp[i] = (double) x[i]; }
        //~ 
        //~ snprintf(buf, 20, "distribution%i.h5", j);
        //~ file_id = H5Fcreate(buf,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
        //~ dims[0] = len;
        //~ H5LTmake_dataset(file_id,"/signal",1,dims,H5T_NATIVE_DOUBLE,tmp);
        //~ dims[0] = 2;
        //~ H5LTmake_dataset(file_id,"/params",1,dims,H5T_NATIVE_DOUBLE,params);
        //~ H5Fclose(file_id);

    }

    *k = j;
}
#endif
