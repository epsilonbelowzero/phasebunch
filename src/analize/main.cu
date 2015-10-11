#include <stdio.h>
#include <stdlib.h>
#include "include/Transform.h"
#include "include/Reader.h"
#include "cuda/Histkernel.h"
#include <assert.h>
#include "include/Output.h"
#include "include/Sanity.h"

int main(int argc, char *argv[]){
	
	assert(argc!= 1);
       	int* hist;
	int hl;       
	double* data;
        double* params = (double*) malloc(sizeof(double)*2);
	hsize_t* dims= (hsize_t*)malloc(sizeof(hsize_t));
	hid_t file; 
	dget(argv,&file,&params,&data,&dims);
	data = (double*) malloc(sizeof(double)*dims[0]);
	readData(&data,&file);
	closeFile(&file);
	transform(data,(int)dims[0]);
	makeHist(&data,&params,(int)dims[0],&hist,1000.0,&hl);		
	check_res(&hist,hl);	
	histOut(&hist,hl);		
	free(data);
	free(params);
	free(dims);
	free(hist);
}


