#include <fftw3.h>

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include <string.h> // for memcpy

int firstMax(double* array, int len) {
	
	double max = 0;
	int i, loc = 0;
	
	for(i = 0; i < len; i++) {
		if( array[i] > max ) {
			max = array[i];
			loc = i;
		}
	}
		
	return loc;	
}

//difference: 	len - number of entries in the array with used values
//				size -total number of entries in the array that can
//					  be written to
//function shifts the array that the first maximum is at array[0]
//and the entries that were before at array[0] - array[1stmax] are
//copied to the end
void shiftFFT(double** array, int len, int size) {
	
	int max = firstMax(*array, len);
	const int s = sizeof(double);
	
	if( size - len >= max ) { //array is big enough to use it as cache
	
		memcpy(&((*array)[len]), *array, max * s);
		memcpy(*array, &((*array)[max]), (len - max) * s);
		memcpy(&((*array)[len - max]), &((*array)[len]), max * s);
		
	} else { //extra cache
		double* cache = (double*) malloc(sizeof(double) * max);
		
		memcpy(cache, *array, max * s);
		memcpy(*array, &((*array)[max]), (len - max) * s);
		memcpy(&((*array)[len - max]), cache, max * s);
		
		free(cache);
	}
	
}

int main(int argc, char** argv) {

    if(argc != 2) {

        printf("Missing file!\n");
        return 1;
        
    }

    printf("Initialising... ");
    fflush(stdout);

    unsigned long long lines = 0;
    unsigned long long i;
    
    //memory for fft
    double* array;
    fftw_complex *out;
    fftw_plan p;

    //hdf5-stuff
    hid_t file_id;
    int d1[1];
    hsize_t dims[1];
    herr_t status;

	//read dimension of data
    file_id = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    H5LTget_dataset_info(file_id,"/signal",dims,NULL,NULL);

	//get memory according to the dimensions, initialise fft
    lines = dims[0];
    array = (double*) fftw_malloc(sizeof(double) * lines);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * floor(lines / 4 + 1));
    p = fftw_plan_dft_r2c_1d(lines/2, array, out, FFTW_MEASURE);
    
    printf("\b\b\b\b (%i records)... ", (int) lines / 2);
    fflush(stdout);

	//read data
    status = H5LTread_dataset_double(file_id,"/signal",array);
    status = H5Fclose(file_id);

	//as only every 2nd element has information about the number of
	//particles passing the detector (the other is the time)
	//copy the important elements to the beginning of the array
    for(i = 0; i < lines / 2; i++) {
		array[i] = array[2*i + 1];
    }
    
    //shift the 1st maximum due to symmetrie-reasons (expected by DFT)
    //the the beginning
    shiftFFT(&array, lines / 2, lines);
    
    printf("Done\nStarting FFT... ");
    fflush(stdout);
    
    fftw_execute(p); //do the ffw
    
    printf("Done\nPrinting... ");
    fflush(stdout);

    //printing...
    hid_t dataspace, dataset;
    file_id = H5Fcreate("fftw.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dim[2] = {(hsize_t) floor(lines/4), 2};
    dataspace = H5Screate_simple(2, dim, NULL);
    dataset = H5Dcreate(file_id, "/fft", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
    //H5LTmake_dataset(file_id, "/fft", 2, dims, H5T_NATIVE_DOUBLE, out);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Fclose(file_id);
  
    fftw_free(p);
    fftw_free(out);
    fftw_free(array);
    printf("Done\n--Finished--\n");
        
    return EXIT_SUCCESS;
}
