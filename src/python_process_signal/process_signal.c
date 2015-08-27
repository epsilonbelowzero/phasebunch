#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <stdio.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <unistd.h>
#include <error.h>

#include <math.h>

#include <omp.h>

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
static int parse_args(PyObject *args, char** filename, int* serial)
{
    if (!PyArg_ParseTuple(args, "s|i", filename, &serial)) {
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
static int readFile(char** filename, long double** params, long double **array, int* lines) {
	
	//hdf5-stuff
    hid_t file;
    hid_t dataset, filespace, memspace;
    hid_t prop;
    int dataDim;//dataDim - dimension of data, i.e. 1-dim (array), 2-dim (2d-array), ...
    hsize_t dims[1];

	if( access(*filename, F_OK) ) {
		PyErr_SetString(PyExc_FileNotFoundError, "File not found");
		return -1;
	}
	if( access(*filename, R_OK) ) {
		PyErr_SetString(PyExc_PermissionError, "File reading permission denied!");
		return -1;
	}

    file = H5Fopen(*filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    if(file < 0) {
		PyErr_SetString(PyExc_IOError, "File Could not be opened");
		return -1;
	}

	dataset = H5Dopen2 (file, "/params", H5P_DEFAULT);
#warning "Fixed params-size in hdf5-file!"
	*params = (long double*) malloc(sizeof(long double) * 6);
	H5Dread(dataset, H5T_NATIVE_LDOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    *params);
    H5Dclose(dataset);                
    
	dataset = H5Dopen2(file, "/signal", H5P_DEFAULT);
	filespace = H5Dget_space (dataset);
	dataDim = H5Sget_simple_extent_ndims (filespace);
	H5Sget_simple_extent_dims (filespace, dims, NULL);
	*lines = (int) dims[0];
	
	prop = H5Dget_create_plist (dataset);

	memspace = H5Screate_simple (dataDim, dims, NULL);
	*array = (long double*) malloc(sizeof(long double) * (*lines));
	
	if( *array == NULL ) {
		PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for array containing the file!");
		return -1;
	}
	
	H5Dread (dataset, H5T_NATIVE_LDOUBLE, memspace, filespace,
                      H5P_DEFAULT, *array);

	H5Pclose (prop);
    H5Dclose (dataset);
    H5Sclose (filespace);
    H5Sclose (memspace);
    H5Fclose(file);
    
    return 0;
}

static int inserting(long double *params, long double **array, int** y, long double** x, int* size, int lines)
{
#warning "Fixed Time-Step size!!!"
    int offset = (int) ((params[0] + 1) * params[5] * 1e9);
    printf("Mem alloc: %Lu Bytes\n", (long long unsigned int) (offset * sizeof(int) * omp_get_max_threads()));
    
    if(offset * sizeof(int) * omp_get_max_threads() >= 1024L*1024L*1024L*2L) {
       PyErr_SetString(PyExc_MemoryError, "More than 2 GB would be allocated!");
       return -1;
    }

    *y = (int*) malloc(offset * sizeof(int));
    for(int i = 0; i < offset; i++) {
        (*y)[i] = 0;
    }
    int* y_t;

#pragma omp parallel
    {
    const int maxThreads = omp_get_num_threads();
    const int nThread = omp_get_thread_num();

    #pragma omp single
    y_t = (int*) malloc(maxThreads * offset * sizeof(int));
    
    for(int i = 0; i < offset; i++) {
        y_t[offset * nThread + i] = 0;
    }

    #pragma omp for
        for(int i = 0; i < lines; i++) {
            y_t[(int) (floorl(fabsl((*array)[i]) * 1e9) + offset * nThread)] += 1;
        }

    #pragma omp for
        for(int i = 0; i < offset; i++) {
            for(int j = 0; j < maxThreads; j++) {
                (*y)[i] += y_t[i + offset * j];
            }
        }
    }
    free(y_t);

    //~ int i = offset - 1;
    //~ for( i = offset - 1; y[i] == 0; i-- ) {}
    *size = offset;
    *y = (int*) realloc(*y, sizeof(int) * (*size));
    
    *x = (long double*) malloc(sizeof(long double) * (*size));
    int k;
#pragma omp parallel for default(none) shared(x, size, params) private(k)
    for( k = 0; k < *size; k += 1) {
        (*x)[k] = k * params[2];
    }
    
    return 0;
}

	//generate numpy-array
	//as python cannot handle c-arrays, it needs to be converted into
	//an object python can handle. numpy has the advantage, that the
	//c-array dont needs to be extracted into a list (which would mean iterating over
	//each element) and an option exists that numpy releases the memory
	//when its no longer needed
static PyObject* buildNumpyArray(int length, int type, void** array) {
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
	int return_code;
	long double* array;//array containing the file-content
	int lines;//number of file-entries
	long double* params;
	int* y;
	long double* x;
	
	//arguments parsed. if an error occurred, an exception is already
	//thrown, and NULL is returned
	printf("Parsing args\n");
	return_code = parse_args(args, &filename, &serial);
	if( return_code < 0 ) {
		return NULL;
	}

	//defines the array which will contain the positions of the particles
	//reads the data from the given file and stores it in the array
	//return NULL if an error occurred, an exception is already thrown
	//in this case
	printf("Reading file\n");
	return_code = readFile(&filename, &params, &array, &lines);
	if( return_code < 0) {
		return NULL;
    }

    printf("Seperating...\n");
    
    int size;
    return_code = inserting(params, &array, &y, &x, &size, lines);
    if( return_code < 0) {
        return NULL;
    }
	free(params);
    free(array);
		
	PyObject* yValues = buildNumpyArray(size, NPY_INT, (void**) &y);
	PyObject* xValues = buildNumpyArray(size, NPY_LONGDOUBLE, (void**) &x);
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
static struct PyMethodDef PyInit_process_signal_methods[] = {{
   "process",
   process,
   METH_VARARGS,
   "actual function doing the stuff"
}};

//Doc: https://docs.python.org/3.1/c-api/module.html#PyModuleDef
//moduleinformation
static PyModuleDef PyInit_process_signal_module = { 
	PyModuleDef_HEAD_INIT,
    "read_sort",
    "Reads H5-File with specific format and sorts the content",
    -1,
    PyInit_process_signal_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

//function which is called when the module is loaded.
PyMODINIT_FUNC
PyInit_process_signal(void)
{
	//create the module with the information given above
	PyObject* m = PyModule_Create(&PyInit_process_signal_module);
	if( m == NULL) {
		return NULL;
	}
	
	import_array(); //initialise numpy - else its functions could not be used
	
	return m;
}
