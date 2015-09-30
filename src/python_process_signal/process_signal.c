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

#include <assert.h>

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
static int parse_args(PyObject *args, char** filename)
{
    if (!PyArg_ParseTuple(args, "s", filename)) {
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
	*params = (long double*) malloc(sizeof(long double) * 2);
	H5Dread(dataset, H5T_NATIVE_LDOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    *params);
    H5Dclose(dataset);

	dataset = H5Dopen2(file, "/signal", H5P_DEFAULT);
	filespace = H5Dget_space (dataset);
	dataDim = H5Sget_simple_extent_ndims (filespace);
	H5Sget_simple_extent_dims (filespace, dims, NULL);
	*lines = (int) dims[0];

	printf("Mem-Alloc (readFile): % .2Lf MB\n", ((long double) sizeof(long double)) * (*lines) / 1024.f / 1024.f);
	*array = (long double*) malloc(sizeof(long double) * (*lines));
	
	if( *array == NULL ) {
		PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for array containing the file!");
		return -1;
	}

	memspace = H5Screate_simple (dataDim, dims, NULL);	
	H5Dread (dataset, H5T_NATIVE_LDOUBLE, memspace, filespace,
                      H5P_DEFAULT, *array);

    H5Dclose (dataset);
    H5Sclose (filespace);
    H5Sclose (memspace);
    H5Fclose(file);
    
    return 0;
}

static long double findMax(long double** array, int size) {
	
	long double maxElement = 0;
	for(int i = 0; i < size; i++) {
		if(fabsl((*array)[i]) > maxElement) {
			maxElement = fabsl((*array)[i]);
		}
	}
	
	return maxElement;
}

static int inserting(long double *params, long double **array, int** y, long double** x, int* size, int lines)
{
#warning "Fixed Time-Step size!!!"
    long double max = findMax(array, lines);
    int offset = (int) (2 * ceil(max / params[0]) + 3),
		halfOffset = (int) (ceil(max / params[0]));
    printf("Mem alloc (Insert): % .2Lf MB\n", ((long double) offset) * sizeof(int) * omp_get_max_threads() / 1024.f / 1024.f);
    
    if( ((long double) offset) * sizeof(int) * omp_get_max_threads() >= 1024.f*1024.f*1024.f*2.f) {
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
	 	/*

		Get an array which is n*offset on 
		in the next step initialize every value with 0
		
		If we would sort everything in one array we
		would get a race condition!So we make a pretty
		long array and every thread gets his own sorting 
		array.(It's very memory consuming but faster!)



		In the last sorting step you'll have to look out: 
		
		At first take the value and take it times 1e9 to 
		get a whole index number! This times half the offset+1
		will give you the right indexnumber. 

		Imagine a array which has one point for each timestep. 
		First we look out where our Index is placed and then
		add half a period to get the right period.


		*/	
		for(int i = 0; i < offset; i++) {
			y_t[offset * nThread + i] = 0;
		}

		#pragma omp for
		for(int i = 0; i < lines; i++) {
			y_t[(int) (floorl((*array)[i] / params[0]) + halfOffset + 1 + offset * nThread)] += 1;
		}

		#pragma omp for
		for(int i = 0; i < offset; i++) {
			for(int j = 0; j < maxThreads; j++) {
				(*y)[i] += y_t[i + offset * j];
			}
		}
    }
    free(y_t);

    *size = offset;
    *x = (long double*) malloc(sizeof(long double) * (*size));
    int k;
#pragma omp parallel for default(none) shared(x, size, params, halfOffset, offset) private(k)
    for( k = 0; k < offset; k ++) {
        (*x)[k] = (k - halfOffset - 1) * params[0];
    }
    
    return 0;
}




void insert2(long double *params, long double **array, int** y, long double** x, int* size, int lines){

	/**


	Initialize the needed memory parameters, 
	a0 ist how many bins you'll need to fill
	the array + 3!
	a1 is half the number of bins you'll need
	(that means how many bins will fit in one amplitude)



	**/

	max = findMax(array,lines);
	long T = (long) 1/params[0]
        int a0 = (int)(2*ceil(max/params[0])+3); 	
	int a1 = (int) (ceil(max/params[0])+3); 
	

        **y = (int*)malloc(sizeof(int)*a0);
	for(int i = 0; i<a0;i++){

	    *y[i]=0;

	} 	
	
        //Now all the Values from array should be sorted into an
	//Bin
	
	 







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
	
	return numpylist;
}

/*
 * Main-function, which is called by the interpreter and the main-program.
 * 
 */
static PyObject *process(PyObject *self, PyObject *args) {
	
	//the parameters and their defaults are set
	char* filename;
	long double* array;//array containing the file-content
	int lines;//number of file-entries
	int return_code;
	long double* params;
	int* y;
	long double* x;
	
	//arguments parsed. if an error occurred, an exception is already
	//thrown, and NULL is returned
	printf("Parsing args\n");
	return_code = parse_args(args, &filename);
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

    printf("Inserting Part 1/2 \n");
    
    int size;
    return_code = inserting(params, &array, &y, &x, &size, lines);
    if( return_code < 0) {
        return NULL;
    }


    #pragma omp parallel for

    for(int i = 0; i < lines; i++){

	array[i]=1/((array[i])+1/params[0]);	

    }

    /*
	Now use insert for the frequency space!

    */
    printf("Inserting Part 2/2 \n");	
    long *x1, *y1;
    params[0]=1/params[0];
    insert2(params,&array,&x1,&y1,&size,lines); 
      

    free(array);
		
	PyObject* yValues = buildNumpyArray(size, NPY_INT, (void**) &y);
	PyObject* xValues = buildNumpyArray(size, NPY_LONGDOUBLE, (void**) &x);
	PyObject* x1Values = buildNumpyArray(size,NPY_LONG,(void**) &x1);
	PyObject* y1Values = buildNumpyArray(size,NPY_LONG,(void**) &y1);
	PyObject* retList = PyList_New(5);

	if(retList == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "Could not create returning list!");
		return NULL;
	}
	
	PyList_SET_ITEM(retList, 0, xValues);
	PyList_SET_ITEM(retList, 1, yValues);
	PyList_SET_ITEM(retList, 2, 
		PyArray_Scalar(
			&(params[1]),
			PyArray_DescrFromType(NPY_LONGDOUBLE),
			PyLong_FromLong(1)
		)
	);
	PyList_SET_ITEM(retList,3,x1Values);
	PyList_SET_ITEM(retList,4,y1Values);
     	
	free(params);

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
