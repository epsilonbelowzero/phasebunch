#include <stdio.h>
#include <stdlib.h>
#include "Reader.h"
#include "Transform.h"
#include "cuda/Sorter.h" // As long as Cuda is on this will work!





int main(int argv, char *argc[]){


/*
Read the HDF5 file and assign the simulation parameters to
data and params!



*/
	
        long double* data;
        long double* params = (long double*) malloc(sizeof(long double)*2);
	hsize_t* dims = (hsize_t*) malloc(sizeof(hsize_t)*1);
	readParams(argc,&params,&dims);	
	data=(long double*) malloc(sizeof(long double)*dims[0]);
	readData(argc,&data);



/*
After IO is done we'll have to transform our data
In this version I won't implement it parallel since 
the Data is small enough and the transformation is one line of code! 

*/
	
	int length = (int) data[0];
	double* data2; 
	castData(&data,&data2,length);	

	

/*
The next step is to sort the array! This can be done by just looking for the
maximum of the absolute value in data!
0. cast the Data to double for Cuda
1. get the Maximum of the Simulation Data
2. With the Maximum we can calculate how many 

*/
	

	double a = findMax(length,&data2);	
	

}




