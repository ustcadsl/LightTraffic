#!/bin/bash

rm mem-size-result.txt

declare -a p=(25 50 100 200)
declare -a b=(100000000 200000000 400000000 800000000)
declare -a r=(8 4 2 1)

plen=${#p[@]}
blen=${#b[@]}

path=$1
dataset=$path/friendster.bcsr
../../pagerank -f $dataset -m -s 72

iteration=5


for (( i=0; i<$iteration; i++ ));
do
    for (( j=0; j<$plen; j++ ));
    do
        for (( k=0; k<$blen; k++ ));
        do
            ../../pagerank -f $dataset -p ${p[$j]} -w ${b[$k]} --runs ${r[$k]} >> mem-size-result.txt
        done
    done
done

cat mem-size-result.txt | grep -o "[0-9|.]* ms" | grep -o "[0-9|.]*" > mem-size-data.txt
