#ifndef PRINT_H
#define PRINT_H

#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#ifndef RANK
#define RANK 1 //data-dimension ( = 1)
#endif

void printInitDistribution(long double** p, int length) {
	double *tmp;
    tmp = (double*) malloc(sizeof(double) * length);
    for(int i=0; i < length; i++) {
		tmp[i] = (double) (*p)[i];
	}
    
    hid_t file_id;
	hsize_t dims[1] = { (long long unsigned int) length };
	
    file_id = H5Fcreate("distribution.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
	H5LTmake_dataset(file_id,"/signal",1,dims,H5T_NATIVE_DOUBLE,tmp);
	H5Fclose(file_id);
    free(tmp);
}

void InitResultFile(hid_t* file, hid_t* dataset, int len) {
	//for memory-reasons, whenever the particle-positions are to be
    //stored, they are directly written into the resulting file.
    //So - a lot of hdf5-stuff to do here
	hsize_t dim[1] = { (long long unsigned int) len }; //length of a chunk, which is the number of particles
	hsize_t maxdim[1] = {H5S_UNLIMITED};
	
    hid_t   dataspace;
    hid_t   prop;
    
    //create file
    *file = H5Fcreate ("result.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);    
    
    //create data-space w/o dimension-limit
    dataspace = H5Screate_simple (RANK, dim, maxdim);

    /* Modify dataset creation properties, i.e. enable chunking  */    
    prop = H5Pcreate (H5P_DATASET_CREATE);
    H5Pset_chunk (prop, RANK, dim);
    /* Create a new dataset within the file using chunk 
       creation properties.  */
    *dataset = H5Dcreate2 (*file, "signal", H5T_NATIVE_LDOUBLE, dataspace,
                         H5P_DEFAULT, prop, H5P_DEFAULT);
                         
    H5Pclose (prop);
    H5Sclose (dataspace);
}

void SaveChunk(hid_t* dataset, int turn, int len, long double beamspeed, long double** x) {
	hid_t filespace, memspace;
	hsize_t dim[1] 		= { (long long unsigned int) len };
	hsize_t offset[1] 	= { (turn-1) * dim[0] };
	hsize_t size[1] 	= { turn * dim[0] };
	
	long double *tmp; tmp = (long double*) malloc(len * sizeof(long double));
	
	H5Dset_extent (*dataset, size);
	/* Select a hyperslab in extended portion of dataset  */
	filespace = H5Dget_space(*dataset);
	H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
								  dim, NULL);  
	/* Define memory space */
	memspace = H5Screate_simple (RANK, dim, NULL);
	
	int i;
	#pragma omp parallel for default(none) private(i) shared(tmp, x, len, beamspeed) if(len > 4)
	for(i = 0; i < len; i++) {
		tmp[i] = (*x)[i] / beamspeed;
	}

	/* Write the data to the extended portion of dataset  */
	H5Dwrite (*dataset, H5T_NATIVE_LDOUBLE, memspace, filespace,
					   H5P_DEFAULT, tmp);
					
	
	H5Sclose (memspace);
	H5Sclose (filespace);
	free(tmp);
}

void FinalizeResultFile(hid_t *dataset, hid_t* file,
	long double dt, long double circumference, long double beamspeed) {
    /* Close resources */
    H5Dclose (*dataset);
    
    hsize_t size[1] = { 2 };
    long double params[] = { dt , circumference / beamspeed};
    H5LTmake_dataset(*file,"/params",1,size,H5T_NATIVE_LDOUBLE,params);
    
    H5Fclose (*file);
}

#endif
