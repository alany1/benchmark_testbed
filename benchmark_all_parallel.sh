#! /bin/bash

############################################################
#
# benchmark_all.sh
# Code to execute benchmark tests
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################

if (( $# == 5 )); then
  for filepath in $1*/; do
    python benchmark_test_parallel.py --poisons_path ${filepath} --dataset $2  --$3 --output $4 --ffcv $5
  done
elif (( $# == 4 )); then
  for filepath in $1*/; do
    python benchmark_test_parallel.py --poisons_path ${filepath} --dataset $2 --output $3 --ffcv $4
  done
else
  echo "Illegal number of arguments."
fi
