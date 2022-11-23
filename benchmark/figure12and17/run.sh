#!/bin/bash

rm -r result
mkdir -p result
iteration=5

make -j

declare -a s=(32 64 128 256 512 1024)
declare -a p=(320 160 80 40 20 10 5)
declare -a method=(opt baseline1)

path=$1
dataset=$path/friendster.bcsr

plen=${#s[@]}
mlen=${#method[@]}

for (( j=0; j<$mlen; j++ ))
do
    for (( i=0; i<$plen; i++ ))
    do
        nsys nvprof ${method[$j]} -f $dataset --length 80 -m -s ${s[$i]} -p ${p[$i]} > result/profile-${method[$j]}-$i-0.txt

        for (( k=1; k<$iteration; k++ ))
        do
            nsys nvprof ${method[$j]} -f $dataset --length 80 -p ${p[$i]} > result/profile-${method[$j]}-$i-$k.txt
        done

        for (( k=0; k<$iteration; k++ ))
        do
            echo "kernel time calls" > result/kernel-${method[$j]}-$i-$k.txt
            cat result/profile-${method[$j]}-$i-$k.txt | grep "void" | awk '{print $9, $2, $3 }' >> result/kernel-${method[$j]}-$i-$k.txt
        done

    done
done

rm report*
