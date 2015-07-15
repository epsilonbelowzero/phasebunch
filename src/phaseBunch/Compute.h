#ifndef COMPUTE_H
#define COMPUTE_H

//~ #include <math.h>
#include <cmath>
#include <stdio.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <assert.h>

#include "Params.h"

//#include <omp.h>

#ifndef SOL
#define SOL 299792458
#endif

long double computeGamma(long double px, long double m) {
    return sqrtl(1 + (px*px) / (m*m));//changed from - to + as E^2 = E0^2 + (pc)^2 and E = gamma * E0
}

long double computeVi(long double pi, long double gamma, long double m) {
    //v_i = p_i * c / (gamma * m)
    return pi / (gamma * m);//removed product with SOL as px stores energy, not energy over c TODO: check this
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
    int len, int *k,
    long double beamspeed, long double circumference,
    long double *freq
) {

    int i,j;
    
    //for memory-reasons, whenever the particle-positions are to be
    //stored, they are directly writtin into the resultant file.
    //So - a lot of hdf5-stuff to do here
	hsize_t dim[1] = { (long long unsigned int) len }; //length of a chunk, which is the number of particles
	hsize_t maxdim[1] = {H5S_UNLIMITED};
	hsize_t offset[1];
	hsize_t size[1] = { 0 };
	const hsize_t RANK = 1; //dimension of data, which is 1
	
	hid_t        file;                          /* handles */
    hid_t        dataspace, dataset;  
    hid_t        filespace, memspace;
    hid_t        prop;   
    herr_t 		status;
    
    //create file
    file = H5Fcreate ("result.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);    
    //create data-space w/o dimension-limit
    dataspace = H5Screate_simple (RANK, dim, maxdim);

    /* Modify dataset creation properties, i.e. enable chunking  */    
    prop = H5Pcreate (H5P_DATASET_CREATE);
    H5Pset_chunk (prop, RANK, dim);
    /* Create a new dataset within the file using chunk 
       creation properties.  */
    dataset = H5Dcreate2 (file, "signal", H5T_NATIVE_DOUBLE, dataspace,
                         H5P_DEFAULT, prop, H5P_DEFAULT);
	
	double *tmp; tmp = (double*) malloc(len * sizeof(double));
    long double t;
    long double h = 1 / ((*freq) * dt);
    for( t = t_start,j = 1; t < t_end - dt; t += dt) {

#pragma omp parallel for default(none) private(i) shared(len, x, px, dt, m, q, t)
        for(i = 0; i < len; i++) {
			updateParticle(t, dt, &(x[i]), &(px[i]), q[i], m[i]);
        }
        
        //check, whether sync-particle passed the detector
        if( t * beamspeed > j * circumference ) {
			//store the current particle-positions, corrected by the current time
			//(the particle's offset to the sync-particles are computed)
			
		    /* Extend the dataset. Dataset becomes 10 x 3  */
			size[0] = size[0]+ dim[0];
			assert(size[0] == j * dim[0]);
			H5Dset_extent (dataset, size);

			/* Select a hyperslab in extended portion of dataset  */
			filespace = H5Dget_space(dataset);
			offset[0] = (j-1) * dim[0];
			H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
										  dim, NULL);  

			/* Define memory space */
			memspace = H5Screate_simple (RANK, dim, NULL);

			#pragma omp parallel for default(none) private(i) shared(tmp, freq, t, px, m, x, len)
			for(i = 0; i < len; i++) {
				tmp[i] = (double) (2 * M_PI / (*freq) * t + computeGamma(px[i], m[i]) * x[i] / computeVi(px[i], computeGamma(px[i], m[i]), m[i]));
			}

			/* Write the data to the extended portion of dataset  */
			H5Dwrite (dataset, H5T_NATIVE_DOUBLE, memspace, filespace,
							   H5P_DEFAULT, tmp);
						
			
			H5Sclose (memspace);
			H5Sclose (filespace);
						
			j++;
		}

    }
    
    /* Close resources */
    printf("dataset\n");
    H5Dclose (dataset);
    printf("prop\n");
    H5Pclose (prop);
    printf("dataspace\n");
    H5Sclose (dataspace);
    printf("file\n");
    H5Fclose (file);

	printf("j = %i\n", j);

    *k = j;
}
#endif
