#################
# Required CMakeVersion
#################

cmake_minimum_required(VERSION 2.8.12)

add_subdirectory("./src/parallelPart" "./build")
add_subdirectory("./src/fft" "./fft")
add_subdirectory("./src/python_process_distribution" "./python_process_distribution")
