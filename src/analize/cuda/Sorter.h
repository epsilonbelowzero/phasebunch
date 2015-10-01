#ifndef SORTER_CUH
#define SORTER_CUH


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







#endif
