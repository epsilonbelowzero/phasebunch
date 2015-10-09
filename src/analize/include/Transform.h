#ifndef TRANSFORM_H
#define TRANSFORM_H



void transform_inv(double** data,int l,double** params){
	
	for(int i = 0; i < l ; i++){
		
		*(data[i]) = 1 / *(data[i]);
		
	}	
	*(params[0]) = 1 / *(params[0]);
}


#endif
