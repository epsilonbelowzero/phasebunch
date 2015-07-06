#!/bin/sh

programName="parallelParts"

echo "$(tput setab 7)                         $(tput sgr0)"
echo "$(tput setab 7)  $(tput sgr0) $(tput bold)$(tput setaf 4)build $programName$(tput sgr0) $(tput setab 7)  $(tput sgr0)"
echo "$(tput setab 7)                         $(tput sgr0)"

tput setaf 3

#check for cmake, make
if [ ! $(which cmake) -o ! $(which make) ]
then
	echo "Either cmake or make could not be found."
	exit 4
fi

#check current pwd - try to get to parallelParts/ (first one)
currentPwd="$(pwd)"
while [ ! -e "CMakeLists.txt" -a "$(pwd)" != "/" ]
do
	cd ..
done

if [ "$(pwd)" = "/" ]
then
	echo "Please go into parallelParts-directory or in a sub-directory of it."
	exit 3
fi

# check whether there is a 'build'-directory. if not, create it and enter
if [ ! -d "./build" ]
then
	mkdir "build"
fi

#check number of cores for later use with make
cores=1
if [ $(which nproc) ]
then
	cores=$(nproc)
elif [ $(which grep) -a -f /proc/cpuinfo ]
then
	cores=$(grep -c ^processor /proc/cpuinfo)
fi

echo "Found number of Cores: ${cores}. Use $(expr ${cores} + 1) jobs for parallel build using make."

# for mac check for gcc version 4.9 (as clang does not support OpenMP), when gcc found, add the option to cmake
addOption=""
if [ $(uname -s) = "Darwin" ]
then
	echo "Detected MAC"
	if [ $(which g++-5) -a $(which gcc-5) ]
	then
		export CC=$(which gcc-5)
		export CXX=$(which g++-5)
		echo "found g++ and gcc version 4.9, add -DCMAKE_CXX_COMPILER=g++-4.9 and -DCMAKE_CC_COMPILER=gcc-4.9 to cmake"
	else
		echo "Could not detect gcc version 4.9, use cmake's default compiler. "
	fi
fi

echo "Running cmake"
tput sgr0

cmake "${addOption}" .

tput setaf 3
echo "Running make"
tput sgr0

make -j$((cores+1)) --no-print-directory

tput setaf 3
echo "Finished"
tput sgr0
