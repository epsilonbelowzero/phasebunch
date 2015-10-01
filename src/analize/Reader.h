#ifndef READER_H
#define READER_H

#include "hdf5.h"
#include "hdf5_hl.h"
#include <stdio.h>



void readParams(char** argc,long double** params,hsize_t** dims ){

	
 	hid_t file; // Create all the necessary attributes to read the file!
       	hid_t dataset;
	hid_t filespace;
	file = H5Fopen(argc[1], H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset = H5Dopen2(file,"/params",H5P_DEFAULT);
	H5Dread(dataset, H5T_NATIVE_LDOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *params);	
	H5Dclose(dataset);
	H5Dopen2(file,"\signal",H5P_DEFAULT);
	filespace = H5Dget_space(dataset);
	int k = (int) H5Sget_simple_extent_ndims(filespace);
	*dims = (hsize_t*) malloc(sizeof(hsize_t)*k);
	H5Sget_simple_extent_dims(filespace,*dims,NULL);
	printf("The dataset dimension is %i \n",k);
	H5Dclose(dataset);			
	H5Fclose(file);		
	
}

void readData(char** argc, long double** data){

	hid_t file;
	hid_t dataset;
	file=H5Fopen(argc[1],H5F_ACC_RDONLY,H5P_DEFAULT);
	dataset = H5Dopen2(file,"/signal",H5P_DEFAULT);
	H5Dread(dataset,H5T_NATIVE_LDOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,*data);
	H5Dclose(dataset);
	H5Fclose(file);






}





#endif
