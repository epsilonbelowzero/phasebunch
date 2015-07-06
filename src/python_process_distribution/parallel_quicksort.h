#ifndef PARALLEL_QUICKSORT
#define PARALLEL_QUICKSORT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//from http://www.oracle.com/technetwork/server-storage/solarisstudio/documentation/oss-openmp-30-wp-183987.pdf
//parallel quicksort, using OpenMP and its task-directive

//defines the pivot-element, sorting it according to quicksort, returning
//the position of the pivot-element
int partition (int p, int r, double *data)
{
	double x = data[p];
	int k = p;
	int l = r + 1;
	double t;

	do
		k++;
	while ((data[k] <= x) && (k < r));
	do
		l--;
	while (data[l] > x);
	
	while (k < l) {
		t = data[k];
		data[k] = data[l];
		data[l] = t;
		do
			k++;
		while (data[k] <= x);
		do
			l--;
		while (data[l] > x);
	}
	t = data[p];
	data[p] = data[l];
	data[l] = t;
	return l;
}

//upon a threshold of number of elements is undercut, its no longer
//recursive but iterative called
void seq_quick_sort (int p, int r, double *data)
{
	if (p < r) {
		int q = partition (p, r, data);
		seq_quick_sort (p, q - 1, data);
		seq_quick_sort (q + 1, r, data);
	}
}
//standard-function. recursive calls
void rec_quick_sort (int p, int r, double *data, int low_limit)
{
	if (p < r) {
		if ((r - p) < low_limit) {//check, whether the number of elements is lower than treshold
			seq_quick_sort (p, r, data);
		}
		else {
			int q = partition (p, r, data);
			//for each call register a new task
			//the tasks are stored as a stack, so each processor requires
			//a new task if it's done with his prior one
			#pragma omp task firstprivate(data, low_limit, r, q)
				rec_quick_sort (p, q - 1, data, low_limit);
			#pragma omp task firstprivate(data, low_limit, r, q)
				rec_quick_sort (q + 1, r, data, low_limit);
		}
	}
}

//entry-point for the mainprogram. array to be sorted, length and the treshold for iterative quicksort
void quick_sort (double *data, int n, int low_limit)
{
#pragma omp parallel
{
	//do only once do the call for the whole array
	#pragma omp single nowait
		rec_quick_sort (0, n - 1, data, low_limit);
}
}

//old mainprogram for the quicksort. please read the source :
//http://www.oracle.com/technetwork/server-storage/solarisstudio/documentation/oss-openmp-30-wp-183987.pdf

/*

int	main (int argc, char *argv[])
{
	int i, n, low_limit;
	float *data;
	
	if (argc != 3) {
		printf ("a.out num_elems low_limit\n");
		return 1;
	}
	
	n = atoi(argv[1]);
	low_limit = atoi(argv[2]);
	*
	* Generate the array.
	*
	data = (float *)malloc (sizeof (float) * n);
	#pragma omp parallel for default(none) shared(data, n) private(i)
	for ( i=0; i<n; i++ ) {
		data[i] = 1.1 * rand() * 5000 / RAND_MAX;
	}
	
	printf ("\nSorting %d numbers using OpenMP tasks...\n\n", n);
	
	struct timespec start, end;
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	par_quick_sort (n - 1, &data[0], low_limit);
	
	clock_gettime(CLOCK_MONOTONIC, &end);
	
	printf("Sorting lasts for %d.%d seconds.\n", end.tv_sec - start.tv_sec, end.tv_nsec - start.tv_nsec);
	
	printf ("Done\n");
	validate_sort (n, &data[0]);
	
	free (data);
	return 0;
}
* */

#endif
