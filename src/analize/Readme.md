As we found out: Our Simulation analysis is very very very limited. 

Test scenarios with relative small initial conditions(dt = 1e-9, t_end=1e-2,length=1e5) create so large hdf5 files which simply aren't possible to analize using python.

We therefore will write an analysis programm for out HDF5 files.



What I'm currently thinking of:



0. input of HDF5 result files



1. Analysis of the resut file



 *  Read the file in chunks

 *  Maybe using Omp/MPI/CUDA reading the file & use the threads to analise the file!

 *  Get one small analised result





2.  Save the analysis file 





# TODO

Feature | Status

-----------|--------------

Find an algorithm which allows to read an HDF5 file/data in chunks by different threads| :heavy_multiplication_x: 

Read a HDF5 File  in chunks |  :heavy_multiplication_x: 

Each thread does the analysis of a chunk | :heavy_multiplication_x: 

Let the result get into one file | :heavy_multiplication_x: 

Save the file | :heavy_multiplication_x: 





# Build status of feature branch 
