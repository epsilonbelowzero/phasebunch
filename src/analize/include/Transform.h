#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <math.h>




void transform(double* data,int l){
	
	for(int i = 0; i < l;i++){
		
		if(data[i] != 0){
			
			data[i]=1.0/(data[i]);
		}
	}
	


}


#endif
