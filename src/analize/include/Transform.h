#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <math.h>




void transform(double* data,int l){
	
	for(int i = 0; i < l;i++){
		
		if(data[i] != 0){
			
			data[i]=1.0/(data[i]);
			printf("data[%i]=%e\n",i,data[i]);
		}
	}
	


}


/*
 *This code causes an unexplainable segfault,
 *maybe post an Issue to NVIDIA
 */
/*
void transform_inv(double* &data,int l,double* &params){
	
	double help=0;
	printf("data[2]=%e \n",*(data[1]));
	for(int i = 0; i < l ; i++){
	printf("Debug %i \n",i);
			
		*(data[i]) = help;
		if(help != 0){
		
			printf("PDebug 1\n");
			help = 1.0/help;
			*(data[i])=help;
			printf("PDebug 3\n");
		}
		printf("PDebug 2 %e\n",help);
		
	}	
	*(params[0]) = 1 / *(params[0]);
}
*/

#endif
