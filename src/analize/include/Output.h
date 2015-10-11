#ifndef OUTPUT_H 
#define OUTPUT_H 

#include <hdf5.h>
#include <hdf5_hl.h>


void histOut(int **res,int l){
	
	hsize_t dims[1] = {(hsize_t)l};
	hid_t file = H5Fcreate("hist.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
	H5LTmake_dataset_int(file,"/hist",1,dims,*res);
}




#endif
