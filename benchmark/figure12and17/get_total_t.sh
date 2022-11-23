#!/bin/bash

declare -a s=(32 64 128 256 512 1024)
iteration=5
arrlen=${#s[@]}

for (( i=0; i<$arrlen; i++ ))
do
    for (( k=0; k<$iteration; k++ ))
    do
        cat result/profile-opt-$i-$k.txt | grep -o "Time: [0-9|.]* ms" | grep -o "[0-9|.]*" >> total-part-size-t.txt
    done
done
