#ifndef SANITY_H
#define SANITY_H


void check_res(int** res,int l){
	
	int k = 0;
	for(int i = 0 ; i < l ; i++){
		if((*res)[i]!= 0){

			k+=(*res)[i];
		}
	}
	printf("Histgramm Integral Result: %i\n",k);

}




#endif
