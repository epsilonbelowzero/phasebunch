#ifndef HISTKERNEL_H
#define HISTKERNEL_H


#define DEBUG(x) printf("Debug %i\n",x);
#define MAX_MINIMUM -2000000000
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


__global__ void Histkernel(int l,int offset,double* data_dev,int* res_dev,double bs,int b){

	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int index = 0;
	int d = gridDim.x*blockDim.x;
	while(tid < l){
		index = floor(data_dev[tid]/bs)+offset/2;
		if(index >=0 && index < offset){
			atomicAdd(&(res_dev[index]),1);	
			if(b)	printf("TID: %i \n",tid);	
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

double findMax_inv(double** data,int l){

	double tempmax = MAX_MINIMUM;
	for(int i = 0; i < l; i++){
		if(*data[i]!= 0){

			if(tempmax < fabs(1/(*data)[i]){

				tempmax = fabs(1/(*data)[i]);
			}
		} 
	return tempmax;
}



void makeHist(double** data,double** params,int l,int** res1,int** res2,double binsize,int* hl){
	
	double* data_dev1;
	double* res_dev1,*res_dev2;
	double max1 = findMax(data,l);
	double max2 = findMax_inv(data,l);
	int l1 =  ceil(max1/dt*2+1);	
	int l2 = ceil(max2/dt*2+1);	

	cudaMalloc((void**)data_dev1,sizeof(double)*l);
	cudaMalloc((void**)res_dev1,sizeof(int)*l1);
	cudaMalloc((void**)res_dev2,sizeof(int)*l2);
	
		
		
}





/*
void makeHist_inv(double**  data, double** params, int l, int** res,double binsize,int* hl){

	double* data_dev;
	int* res_dev;
	double max1 = findMax(data,l);
	printf("The maximum 2 is: %e\n",max1);
	int offset = (int)(2*ceil(max1/binsize)+1);
 	assert(offset);
	printf("Offset 2 is: %i \n",offset);
 	*res =(int*)malloc(sizeof(int)*offset);
	int count = 0;
	cudaGetDeviceCount(&count);
	printf("count=%i \n",count);
	assert(count);	
	cudaMalloc((void**)&data_dev,sizeof(double)*l);
	cudaMalloc((void**)&res_dev,sizeof(int)*offset);
	cudaMemcpy(data_dev,(*data),l*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(res_dev,(*res),offset*sizeof(int),cudaMemcpyHostToDevice);
	dim3 bpg = dim3(l/32);//Blocks per grid
	dim3 tpb = dim3(32);//Threads per bock
	Histkernel<<<bpg,tpb>>>(l,offset,data_dev,res_dev,binsize,0);
	cudaMemcpy((*res),res_dev,offset*sizeof(int),cudaMemcpyDeviceToHost);	
	*hl = offset;
	cudaFree(data_dev);
	cudaFree(res_dev);

}

void makeHist(double**data,double** params,int l, int **res,int* hl){

	double* data_dev;
	int* res_dev; 
	double max1 =  findMax(data,l);
	int offset = (int) ((2*ceil(max1/(*params)[0]))+1);
	assert(offset);
	printf("Offset 1 is %i \n",offset);
	*res = (int*) malloc(sizeof(int)*offset);
	int count = 0;
	cudaGetDeviceCount(&count);
	printf("count=%i \n",count);
	assert(count);	
	cudaMalloc((void**)&data_dev,sizeof(double)*l);
	cudaMalloc((void**)&res_dev,sizeof(int)*offset);
	cudaMemcpy(data_dev,(*data),sizeof(double)*l,cudaMemcpyHostToDevice);
	cudaMemcpy(res_dev,(*res),sizeof(int)*offset,cudaMemcpyHostToDevice);
	dim3 bpg = dim3(l/32);
	dim3 tpb = dim3(32);
	Histkernel<<<bpg,tpb>>>(l,offset,data_dev,res_dev,(*params)[0],0);
	cudaMemcpy((*res),res_dev,sizeof(int)*offset,cudaMemcpyDeviceToHost);
	*hl = offset;
	cudaFree(data_dev);
	cudaFree(res_dev);
	*res = (int*) malloc(sizeof(int));
	(*res)[0] = 0;
	*hl = 1;
}
*/
#endif
