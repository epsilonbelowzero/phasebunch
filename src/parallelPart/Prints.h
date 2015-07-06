#ifndef PRINT_H
#define PRINT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#define BUF 100

void truncateFile() {
    FILE *f = fopen("results.txt", "w+");
    fclose(f);
}

//not used
void print(long double t,
    long double x[], long double y[], long double z[],
    long double px[], long double py[], long double pz[],
    int len
) {
    FILE *f = fopen("results.txt", "a+");

    int i;
    char buf[BUF];
    char *buffer = (char*) malloc(sizeof(char) * len * BUF);
    strcpy(buffer, "");
    for(i = 0; i < len; i++) {
        snprintf(buf, 100, "%014.10Lf %014.10Lf %014.10Lf ", x[i], y[i], z[i]);
        strcat(buffer, buf);
    }
    strcat(buffer, "\n");
    fwrite( buffer, sizeof(char), strlen(buffer), f);

    free(buffer);
    fclose(f);
}

void print_array(   long double t_start, long double t_end, long double dt,
                    int length, int printEveryNthTimeStep,
                    long double ***result
) {

    FILE *f = fopen("results.txt", "a+");

    int i,j;
    long double t;
    char buf[BUF];
    char *buffer = (char*) malloc(sizeof(char) * length * BUF);

    for( t = t_start,j = 0; t < t_end - dt; t += printEveryNthTimeStep * dt, j++)  {

        strcpy(buffer, "");

        for(i = 0; i < length; i++) {

            snprintf(buf, 100, "%014.10Lf ", (*result)[j][i]);
            strcat(buffer, buf);

        }

        strcat(buffer, "\n");
        fwrite( buffer, sizeof(char), strlen(buffer), f);
    }

    free(buffer);
    fclose(f);

}

void truncate_signals(){

    FILE *f = fopen("signals.txt", "w+");
    printf("signals.txt created \n");
    fclose(f);

}





void print_signal(long double **sign, int length, int k ){

	double* conv_sign =(double*) malloc(sizeof(double)*k*2);

	int i;
#pragma omp parallel for default(none) private(i) shared(k, conv_sign, sign)
	for(i = 0; i < 2*k; i++){

		conv_sign[i]=(double) (*sign)[i];

	}


	hid_t file_id;
	hsize_t dims[1];
	herr_t status;
	dims[0] = 2*k;
	file_id = H5Fcreate("signal.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
	status = H5LTmake_dataset(file_id,"/signal",1,dims,H5T_NATIVE_DOUBLE,conv_sign);
	status = H5Fclose(file_id);

	free(conv_sign);




}







#endif
