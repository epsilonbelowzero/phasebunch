#ifndef OUTPUT_H 
#define OUTPUT_H 

#include <hdf5.h>
#include <hdf5_hl.h>


void histOut(int **hist1,int **hist2,int hl1,int hl2){
	hsize_t dims2[1] = {(hsize_t) hl2};
	hsize_t dims1[1] = {(hsize_t) hl1};
	hid_t file = H5Fcreate("hist.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
	H5LTmake_dataset_int(file,"/hist_inv",1,dims2,*hist2);
	H5LTmake_dataset_int(file,"/hist",1,dims1,*hist1);
	H5Fclose(file);
}




#endif
