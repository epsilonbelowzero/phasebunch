#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <stdio.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <unistd.h>

#include "parallel_quicksort.h"

//Doc: https://docs.python.org/3.1/c-api/structures.html#PyMethodDef

/*
 * The arguments from the call in the appropriate python file results
 * in a list of args, a PyObject (called args). Here the arguments
 * are extracted and converted into c-datatypes.
 * "s|idi" means a obligate string-argument, the others do not needed to
 * be set. i - integer, d - double
 * 
 */
static int parse_args(PyObject *args, char** filename, int* serial, double* interval_length, int *interval_count)
{
    if (!PyArg_ParseTuple(args, "s|idi", filename, interval_count, serial, interval_length)) {
        PyErr_SetString(PyExc_ValueError, "Couldn't parse arguments.");
        return -1;
	}
	
	return 0;
}

/*
 * This function reads the file given in the arguments, and stores
 * the data of the format given in Init.h of the mainprogram in the
 * argument array, also sets the number of entries in the array (lines).
 * 
 * If the file is not found, or if its not readable or another error
 * occurs, python exceptions are thrown.
 * 
 */
static int readFile(char** filename, double **array, int* lines) {
	
	//hdf5-stuff
    hid_t file_id;
    //~ int d1[1];
    hsize_t dims[1];
    //~ herr_t status;

	if( access(*filename, F_OK) ) {
		PyErr_SetString(PyExc_FileNotFoundError, "File not found");
		return -1;
	}
	if( access(*filename, R_OK) ) {
		PyErr_SetString(PyExc_PermissionError, "File reading permission denied!");
		return -1;
	}

    file_id = H5Fopen(*filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    if(file_id < 0) {
		PyErr_SetString(PyExc_IOError, "File Could not be opened");
		return -1;
	}
    
    H5LTget_dataset_info(file_id,"/signal",dims,NULL,NULL);

    *lines = (int) dims[0];
    *array = (double*) malloc(sizeof(double) * (*lines));

    H5LTread_dataset_double(file_id,"/signal",*array);
    H5Fclose(file_id);
    
    return 0;
}

/*
 * This function counts the number of particles which would belong in
 * a sub-interval of [0, length], storese the number in array result.
 * count is the length of the result number and the number of subintervals,
 * data is the position data of the particles with length len (the number
 * of particles), length denotes the interval-length (usually 2*pi).
 * Note, that the array data is sorted!
 */
static int insertInIntervals(double* data, int len, double length, double intervalStart, int* count, int** result, double** x) {
	
	if( length <= 0 ) {
		
		if( data[len - 1] == data[0] ) { //all entries are equal
			*count = 3;
			
			*result = (int*) malloc(sizeof(int) * (*count));
			*x = (double*) malloc(sizeof(double) * (*count));
			
			(*x)[0] = data[0] - 1;
			(*x)[1] = data[0];
			(*x)[2] = data[0] + 1;
			
			(*result)[0] = 0;
			(*result)[1] = len;
			(*result)[2] = 0;
			
			return 0;
		}
		else {
			length = (data[len - 1] - data[0]) / (*count - 2);
			
			printf("%lf - %lf / (%i - 2) = %lf\n", data[len - 1], data[0], *count, length);
		}
	}
	
	*result = (int*) malloc(sizeof(int) * (*count));
	*x = (double*) malloc(sizeof(double) * (*count));
	intervalStart = data[0] - length;
	
#pragma omp parallel sections
{
	
	#pragma omp section
	{
		for(int i = 0; i < *count; i++) {
			(*x)[i] = length * i + intervalStart;
		}
	}
	
	#pragma omp section
	{
		int k = 0, //caches the position of the last particle of the lower subinterval
			j;
		
		for(int i = 0; i < *count; i++) {	
			j = k;
			while(j < len && data[j] < (i + 1) * length + intervalStart) { j++; }
			
			(*result)[i] = j - k;
			k = j;
		}
	}
}

	
	return 0;
}

static PyObject* buildNumpyArray(int length, int type, void** array) {
	//generate numpy-array
	//as python cannot handle c-arrays, it needs to be converted into
	//an object python can handle. numpy has the advantage, that the
	//c-array dont needs to be extracted into a list (which would mean iterating over
	//each element) and an option exists that numpy releases the memory
	//when its no longer needed
	int nd = 1; //1 dimensional
	npy_intp dims[nd];
	dims[0] = length;//length of the dimension
	
	//create numpy array.
	//first argument is firm, number of array-dimensions, length of each array-dimension, data-type, NULL, the array to convert, 0, c-array / it can be written, NULL
	PyObject* numpylist = PyArray_New(&PyArray_Type, nd, dims, type, NULL, (*array), 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL);
	
	if( numpylist == NULL ) {
		return NULL;
	}
	//mark the array as owned, so the memory is released after its no longer needed
	//~ PyArray_ENABLEFLAGS(numpylist, NPY_ARRAY_OWNDATA);
	
	return numpylist;
}

/*
 * Main-function, which is called by the interpreter and the main-program.
 * 
 */
static PyObject *process(PyObject *self, PyObject *args) {
	
	//the parameters and their defaults are set
	char* filename;
	int serial = 1000;
	double length = -1;//tell with negativ value to use max/min value and use count steps between
	int count = 1000;
	int return_code;
	double* array;//array containing the file-content
	int lines;//number of file-entries
	int* result;//array containing the histogramm
	double* x;//to result corresponding x-values
	double intervalStart = 0; //Offset, if the intervals that the particles are inserted to should not start with zero
	
	//arguments parsed. if an error occurred, an exception is already
	//thrown, and NULL is returned
	printf("Parsing args\n");
	return_code = parse_args(args, &filename, &serial, &length, &count);
	if( return_code < 0 ) {
		return NULL;
	}
	
	printf("%i intervalls\n", count);

	//defines the array which will contain the positions of the particles
	//reads the data from the given file and stores it in the array
	//return NULL if an error occurred, an exception is already thrown
	//in this case
	printf("Reading file\n");
	return_code = readFile(&filename, &array, &lines);
	if( return_code < 0) {
		return NULL;
	}
	
	//sorts the array. serial is a treshold where no longer is
	//recursive but iterative sorted
	printf("Sorting\n");
	quick_sort(array, lines, serial);
	
	//counts the number of particles which belong in each subinterval,
	//stores the result in array result
	printf("Seperating\n");
	insertInIntervals(array, lines, length, intervalStart, &count, &result, &x);
	free(array); //position data of the particles are no longer needed
	
	PyObject* yValues = buildNumpyArray(count, NPY_INT, (void**) &result);
	PyObject* xValues = buildNumpyArray(count, NPY_DOUBLE, (void**) &x);
	PyObject* retList = PyList_New(2);
	
	if(retList == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "Could not create returning list!");
		return NULL;
	}
	
	PyList_SET_ITEM(retList, 0, xValues);
	PyList_SET_ITEM(retList, 1, yValues);
	
	return retList;
}
//array of structs of information of each funtion callable of the interpreter.
static struct PyMethodDef PyInit_process_distribution_methods[] = {{
   "process",
   process,
   METH_VARARGS,
   "actual function doing the stuff"
}};

//Doc: https://docs.python.org/3.1/c-api/module.html#PyModuleDef
//moduleinformation
static PyModuleDef PyInit_process_distribution_module = { 
	PyModuleDef_HEAD_INIT,
    "read_sort",
    "Reads H5-File with specific format and sorts the content",
    -1,
    PyInit_process_distribution_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

//function which is called when the module is loaded.
PyMODINIT_FUNC
PyInit_process_distribution(void)
{
	//create the module with the information given above
	PyObject* m = PyModule_Create(&PyInit_process_distribution_module);
	if( m == NULL) {
		return NULL;
	}
	
	import_array(); //initialise numpy - else its functions could not be used
	
	return m;
}
