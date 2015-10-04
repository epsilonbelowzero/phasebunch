#ifndef SORTER_CUH
#define SORTER_CUH
#include <stdlib.h>




__global__ void histkernel(double* data_dev,int* res_dev,int l,int l2,int l3,double bin){

	int index = 0;
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	int foo = blockDim.x*gridDim.x;

	while(i < l){
 	 	index =(int) floor(data_dev[i]/bin+l3);
		atomicAdd(&res_dev[index],1);		
		i+=foo;	
	}




}




double abs1(double v){
	if(v<0){return -1*v;}
	else{ return v;}
}




double findMax(int l,double** data){

	//double* dev_data;
	//cudaMalloc((void**)&dev_data,l*sizeof(double));
	//cudaMemcpy(dev_data,*data,l*sizeof(double),cudaMemcpyHostToDevice);	
	double tempMax=0;
	
	for(int i = 0; i < l;i++){

		if(tempMax<abs1((*data)[i])){

			tempMax = abs1((*data)[i]);
			printf("This is Tempmax: %e\n",tempMax);
		}
	}

	return tempMax;
}


void makeHist(double** data, int** res, double** p,double a,int l,int* l2){
	
	
	
	/*
	 * 1. get an arbritrary bin, then set the lengths(equals the offset/halfoffset in process_signal)
	 * and then initialize the result array for the histogram!
	 * 2. Then initialize the Cuda memory and pointers
	 *
	 * */
	*(p[0])=100; //in s^-1
	*l2 =  (int)(2*a/(*p[0]));
	printf("dt: %e \n",a/(*p[0]));
	int l3 =  (int)(a/(*p[0]));
	*res = (int*) malloc(sizeof(double)*(*l2));
	double *data_dev;
	int    *res_dev;
	

	/*
	 *3. call the Kernel,save result in *res  and free the memory! 
	 *
	 *
	 *
	 *
	 */


	cudaMalloc((void**)&data_dev,l*sizeof(double));
	cudaMemcpy(data_dev,*data,l*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&res_dev,(*l2)*sizeof(int));
	cudaMemcpy(res_dev,(*res),(*l2)*sizeof(int),cudaMemcpyHostToDevice);
	cudaDeviceProp prop;
	int blocks = prop.multiProcessorCount;
	histkernel<<<blocks*2,256>>>(data_dev,res_dev,l,*l2,l3,*(p[0]));
	cudaMemcpy((*res),res_dev,(*l2)*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(data_dev);
	cudaFree(res_dev);


}





#endif
