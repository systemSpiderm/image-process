#!/bin/bash
#

mpic++ -O0 -fopenmp -mavx2 -o image_process image_process.cpp

echo "Please enter the value of the seed:"

while true; do
    read seed

    # 检查输入是否合法
    if [[ "$seed" =~ ^[1-9][0-9]*|0$ ]]; then
        echo "Using seed: $seed"
        break
    else
        echo "Invalid input. Please enter a positive integer."
    fi

done

mpirun -np 4 image_process $seed
