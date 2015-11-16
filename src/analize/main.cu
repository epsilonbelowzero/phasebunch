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
       	int* hist1,*hist2;
	int hl1,hl2;
 	double binsize = 1e4;	
	double* data;
        double* params = (double*) malloc(sizeof(double)*2);
	hsize_t* dims= (hsize_t*)malloc(sizeof(hsize_t));
	hid_t file; 
	
	dget(argv,&file,&params,&data,&dims);
	data = (double*) malloc(sizeof(double)*dims[0]);
	readData(&data,&file);
	closeFile(&file);
	
	makeHist();

	check_res(&hist1,hl1);
	check_res(&hist2,hl2);	
	histOut(&hist1,&hist2,hl1,hl2);		
	free(data);
	free(params);
	free(dims);
	free(hist1);
	free(hist2);
}


