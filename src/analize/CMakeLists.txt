#######################
# Look for the right Cmake version
#######################


cmake_minimum_required(VERSION 3.0.0)




#######################
#Project Stuff
#######################

project(analizer)




######################
#Use some awesome flags
#####################



####################
# Find Cuda
####################





###################
#Find OpenMp
###################
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")


###################
#CUDA
##################
find_package(CUDA REQUIRED)


###################
#Find HDF5 Libs
###################
find_package(HDF5 COMPONENTS HL REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HDF5_LIBRARIES})

#############
#Setup the rest
#############


file(GLOB_RECURSE SRC_FILES "*.cu")

cuda_add_executable(analizer ${SRC_FILES})



target_link_libraries(analizer ${OpenMP_CXX_FLAGS} ${LIBS})
