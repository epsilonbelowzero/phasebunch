#################
# Required CMakeVersion
#################

cmake_minimum_required(VERSION 2.8.12)

add_subdirectory("./src/phaseBunch" "./build/phaseBunch")
#~ add_subdirectory("./src/fft" "./build/fft")
add_subdirectory("./src/python_process_distribution" "./build/python_process_distribution")
add_subdirectory("./src/python_process_signal" "./build/python_process_signal")
add_subdirectory("./src/analysis" "./build")
