#ifndef HISTKERNEL_H
#define HISTKERNEL_H


#define DEBUG(x) printf("Debug %i\n",x);

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

__global__ void Histkernel(int l,int offset,double* data_dev,int* res_dev,double bs){

	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int index = 0;
	int d = gridDim.x*blockDim.x;
	while(tid < l){
		index = floor(data_dev[tid]/bs)+offset/2;
		if(index >=0 && index < offset){

			atomicAdd(&(res_dev[index]),1);	
		
		}
		else{

		  	printf("Error 1: Bad index, index=%i\n",index);
			printf("Error 2: Bad Value %e\n",data_dev[tid]);
		}	
		tid += d;	
	}	
}



double findMax(double** data,int l){

	double tempmax = -1e4;
	for(int i = 0; i<l;i++){

		if(tempmax < fabs((*data)[i])){
			tempmax = fabs((*data)[i]);
		}
	}

	return tempmax;
}


void makeHist(double**  data, double** params, int l, int** res,double binsize,int* hl){

	double* data_dev;
	int* res_dev;
	double max1 = findMax(data,l);
	printf("The maximum is: %e\n",max1);
	int offset = (int)(2*ceil(max1/binsize)+1);
 	assert(offset);
	printf("Offset is: %i \n",offset);
 	*res =(int*)malloc(sizeof(int)*offset);
	int count = 0;
	cudaGetDeviceCount(&count);
	printf("count=%i \n",count);
	cudaMalloc((void**)&data_dev,sizeof(double)*l);
	cudaMalloc((void**)&res_dev,sizeof(int)*offset);
	cudaMemcpy(data_dev,(*data),l*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(res_dev,(*res),offset*sizeof(int),cudaMemcpyHostToDevice);
	dim3 bpg = dim3(l/32);//Blocks per grid
	dim3 tpb = dim3(32);//Threads per bock
	Histkernel<<<bpg,tpb>>>(l,offset,data_dev,res_dev,binsize);
	cudaMemcpy((*res),res_dev,offset*sizeof(int),cudaMemcpyDeviceToHost);	
	*hl = offset;
	cudaFree(data_dev);
	cudaFree(res_dev);

}






#endif
