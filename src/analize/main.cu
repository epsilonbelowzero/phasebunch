#include <stdio.h>
#include <stdlib.h>
#include "Reader.h"
#include "include/Transform.h"
#include "cuda/Sorter.h"	
#include "include/Reader.h"



int main(int argc, char *argv[]){

        double* data;
        double* params = (double*) malloc(sizeof(double)*2);
	hsize_t* dims= (hsize_t*)malloc(sizeof(hsize_t));
	hid_t file; 
	int* hist; 
	dget(argv,&file,&params,&data,&dims);

	transform_inv(&data,dims[0],&params);	

	free(data);
	free(params);
	free(dims);
	free(hist);
}


