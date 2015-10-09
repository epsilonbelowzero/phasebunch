#ifndef READER_H
#define READER_H

/*This Header represents  an adapter to decouple the HDF5 Api from the IO process,
 *if you aren't familiar with this see: 
 *https://en.wikipedia.org/wiki/Adapter_pattern 
 *
 */

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdlib.h"

void openFile(hid_t* handle,char** argv){
	printf("%s \n",argv[1]);
	*handle = H5Fopen(argv[1],H5F_ACC_RDWR,H5P_DEFAULT);	
}

void closeFile(hid_t* handle){

	H5Fclose(*handle);
}

void readParams(double** params,hid_t* handle){

	H5LTread_dataset_double(*handle,"/params",*params);
}

void getDims(hsize_t** dims,hid_t *handle){

	H5LTget_dataset_info(*handle,"/signal",*dims,NULL,NULL);
}

void readData(double** data,hid_t* handle){

	H5LTread_dataset_double(*handle,"/signal",*data);
}

void dget(char* argv[],hid_t* handle,double** params,double** data,hsize_t** dims){

	openFile(handle,argv);
	readParams(params,handle);
	getDims(dims,handle);
	printf("dims[0]= %i\n",*dims[0]);
	*data = (double*) malloc(sizeof(double)*(*(dims[0])));
	readData(data,handle);
	closeFile(handle);
}


#endif
