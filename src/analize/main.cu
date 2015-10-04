#include <stdio.h>
#include <stdlib.h>
#include "Reader.h"
#include "Transform.h"
#include "cuda/Sorter.h"	




int main(int argv, char *argc[]){


/*
Read the HDF5 file and assign the simulation parameters to
data and params!



*/
	
        long double* data;
        long double* params = (long double*) malloc(sizeof(long double)*2);
	hsize_t* dims;
	readParams(argc,&params,&dims);		
	int length = (int) dims[0];
	printf("The length is: %i\n",length);
	data=(long double*) malloc(sizeof(long double)*(dims[0]));
	readData(argc,&data);
	printf("Debug 0\n");




/*
After IO is done we'll have to transform our data
In this version I won't implement it parallel since 
the Data is small enough and the transformation is one line of code! 

*/
	

	double* data2;
        double* params2;	
	printf("length in this scope %i\n",length);
	castData(&data,&data2,length);
	castData(&params,&params2,2);
	printf("Debug 1\n");	
	Transform_inv(&data2,length);	
	Transform_inv(&params2,2);
	double a = findMax(length,&data2);	
	printf("Debug 2\n");	
	printf("The maximum is %e \n",a);	
	printf("The inverse of this is %e \n",1/a);
/*
The next step is to sort the array! This can be done by just looking for the
maximum of the absolute value in data!
0. cast the Data to double for Cuda
1. get the Maximum of the Simulation Data
2. With the Maximum we can calculate how many 

*/




	int* res;
        int k;

  	makeHist(&data2,&res,&params2,a,length,&k);
	printf("k:%i \n",k);	
	saveHist(&res,k);


free(params);
free(dims);
free(data);

	

}




