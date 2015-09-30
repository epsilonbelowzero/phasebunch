#######################
# Look for the right Cmake version
#######################


cmake_minimum_required(VERSION 3.0.0)




#######################
#Project Stuff
#######################

project(analizer)


###############
#Try to find Cuda!
##############
find_package(CUDA)



if(NOT ${CUDA_FOUND})

	###################
	#Find OpenMp
	###################
	find_package(OpenMP REQUIRED)
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
	message("-- Using OpenMP!")

endif(NOT ${CUDA_FOUND})






###################
#Find HDF5 Libs
###################
find_package(HDF5 COMPONENTS HL REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HDF5_LIBRARIES})

#############
#Setup the rest
#############

if(${CUDA_FOUND})

	message("-- Using CUDA")
	include_directories("${PROJECT_SOURCE_DIR}/cuda")
	file(GLOB SRC_FILES "*.cu")
 	cuda_add_executable(analizer ${SRC_FILES})
	target_link_libraries(analizer ${LIBS} )
	

elseif(NOT ${CUDA_FOUND})

	message("-- Using OpenMP instead of CUDA")
	include_directories("${PROJECT_SOURCE_DIR}/include")
	file(GLOB SRC_FILES "*.cpp")
	add_executable(analizer ${SRC_FILES})
	target_link_libraries(analizer ${OPENMP_CXX_FLAGS} ${LIBS})
	message("Source files are: ${SRC_FILES}")	


endif(${CUDA_FOUND})




