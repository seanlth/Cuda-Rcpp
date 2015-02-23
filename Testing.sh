#!/bin/sh

for (( i=10; i<=1000; i+=20 ))
do
    ./R_CUDA $i
done

