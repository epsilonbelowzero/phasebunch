#!/bin/bash
#PBS -q laser
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=64
#PBS -d .
#PBS -o stdout_1
#PBS -e stderr_1

source module_setup

echo "Start..."

build/parallelparts

echo "End..."
