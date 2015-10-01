#ifndef TRANSFORM_H
#define TRANSFORM_H



void castData(long double** data,double** data2,int l){
	
	*data2=(double*) malloc(sizeof(double)*l);
	for(int i = 0; i < l; i++){
	////	printf("@step %i \n",i);
		(*data2)[i]=(double) (*data)[i];	
	}

}

void Transform_inv(double** data,int l){
/*
This section can be parallelized easy but that 
will be done in another version!


*/
	for(int i = 0; i < l; i++){

		(*data)[i]=1/((*data)[i]);

	}

}


#endif
