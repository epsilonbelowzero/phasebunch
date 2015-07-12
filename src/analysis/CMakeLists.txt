#######
#
#######

project(analysis)

#just copy the scripts to the build-directory

file(GLOB SRC "*.py")

foreach(FILE IN LISTS SRC)
	configure_file( ${FILE} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach()
