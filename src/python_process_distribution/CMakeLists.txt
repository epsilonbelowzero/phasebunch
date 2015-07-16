#################
# Required CMakeVersion
#################

cmake_minimum_required(VERSION 2.8.12)

#################
# Project
#################

project(process_distribution)

set(EXTRA_COMPILE_ARGS)

list(APPEND EXTRA_COMPILE_ARGS "-std=c11")

###############
# Files
###############

#collect the files which are needed to build the library
file(GLOB FILES "*.[hc]")
file(GLOB COMPILE_FILES "*.[c]")

#################
# Find Python
#################

#check for python-interpreter and the Python.h-file, as its cannot be
#build w/o them
find_package(PythonLibs 3)
find_package(PythonInterp 3)

################
#Find HDF5 components
###############

find_package(HDF5 COMPONENTS HL)

##########
#Find OpenMP
#########
find_package(OpenMP)

##########
# Configure building python-script:
#############

#create a list in pythons format for each nessecary extra argument
set(LINK_LIST)
foreach(I IN LISTS HDF5_HL_LIBRARY_NAMES_INIT OpenMP_CXX_FLAGS HDF5_LIBRARY_DIRS) #for each neccesary lib-flags
	string(REGEX MATCH "^-" OUTPUT "${I}") #check, whether its already leaded by a "-"
	if(OUTPUT) # leading "-" so no -l is nessecary
		set(LINK_LIST "${LINK_LIST},'${I}'") #add it w/o a "-l"
	else()
        string(REGEX MATCH "/" OUTPUT2 "${I}") #check, whether its already leaded by a "-"
        if(OUTPUT2)
            set(LINK_LIST "${LINK_LIST},'-L${I}'")#add "-L"-option
        else()
            set(LINK_LIST "${LINK_LIST},'-l${I}'")#add "-l"-option
        endif()
	endif()
endforeach()
string(REGEX REPLACE "^[ ]*," "" LINK_LIST "${LINK_LIST}") #remove the leading whitespaces and comma

#same for compile-arguments
set(COMPILE_LIST)
foreach(I IN LISTS OpenMP_CXX_FLAGS HDF5_INCLUDE_DIRS EXTRA_COMPILE_ARGS)
	string(REGEX MATCH "^-" OUTPUT "${I}") #check, whether its already leaded by a "-"
	if(OUTPUT) # leading "-" so no -l is nessecary
		set(COMPILE_LIST "${COMPILE_LIST},'${I}'") #add it w/o a "-l"
	else()
		set(COMPILE_LIST "${COMPILE_LIST},'-I${I}'")#add "-l"-option
	endif()
endforeach()
string(REGEX REPLACE "^[ ]*," "" COMPILE_LIST "${COMPILE_LIST}")

#and source-files
set(COMPILE_FILE_LIST)
foreach(I IN LISTS COMPILE_FILES)
	string(REGEX REPLACE "^.*/" "" I "${I}")
	set(COMPILE_FILE_LIST "${COMPILE_FILE_LIST}, '${I}'")
endforeach()
string(REGEX REPLACE "^[ ]*," "" COMPILE_FILE_LIST "${COMPILE_FILE_LIST}")

configure_file("setup.py.cfg" "setup.py") #replace the variables

###########
# Copy files to build-dir
###########
#neccessary due to relative paths

foreach(I IN LISTS FILES)
	configure_file( ${I} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach()

###########
# Build (via python skript)
###########

file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/build") #remove pror build if its exists
string(REGEX REPLACE "^(.*)/.*" "\\1" TARGET "${CMAKE_CURRENT_BINARY_DIR}") #set target-path
#custom target. call the python-skript and something arround
add_custom_target(process_distribution ALL 
	COMMAND ${PYTHON_EXECUTABLE} setup.py -q build
	COMMAND find . -name "*.so" -exec mv {} ${TARGET} "\;"  #copy to build-dir
	DEPENDS ${FILES} #build the project via calling the python-skript, -q -> quiet output
)
