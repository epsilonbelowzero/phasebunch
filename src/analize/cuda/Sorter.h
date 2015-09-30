#ifndef SORTER_CUH
#define SORTER_CUH


double abs(double v){
	if(v<0)return -1*v;

}




double findMax(int l,double** data){

	//double* dev_data;
	//cudaMalloc((void**)&dev_data,l*sizeof(double));
	//cudaMemcpy(dev_data,*data,l*sizeof(double),cudaMemcpyHostToDevice);	
	int tempMax=0;
	
	for(int i = 0; i < l;i++){

		if(tempMax<abs((*data[i]))){

			tempMax = (*data[i]);

		}


	}

	return tempMax;


}







#endif
