#include <stdio.h>
#include "Reader.h"

int main(int argc, char** argv){


	if(argv == NULL){

		printf("Error! You forgot the File!\n");
		return 0;


	}
	double* data;
	read(argv,&data);
	printf("Data contains: %e \n",data[100]);



}
