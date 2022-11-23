#!/bin/bash

iteration=5

declare -a p=(25 50 100 200)
declare -a b=(100000000 200000000 400000000 800000000)

path=$1
dataset=$path/friendster.bcsr

rm mem-size-result.txt

../../pagerank -f $dataset -m -s 72

plen=${#p[@]}
blen=${#b[@]}

for (( i=0; i<$iteration; i++ ));
do
    for (( j=0; j<$plen; j++ ));
    do
        for (( k=0; k<$blen; k++ ));
        do
            ../../pagerank -f $dataset -w 800000000 -p ${p[$j]} -b ${b[$k]} >> mem-size-result.txt
        done
    done
done

cat mem-size-result.txt | grep -o "[0-9|.]* ms" | grep -o "[0-9|.]*" > mem-size-data.txt
