#!/bin/bash


rm -r result
mkdir result

make -j

declare -a app=(pagerank-opt pagerank-base pagerank-asyn pagerank-fifo)
declare -a parts=(150 100 50 25)

dataset=friendster
source=3142233
partsize=72

step=80
iteration=5

alen=${#app[@]}
plen=${#parts[@]}

path=$1
./pagerank-opt -f $path/$dataset.bcsr -w 100000000 --source $source -m -s $partsize

for (( k=0; k<$plen; k++ ))
do

    for (( j=0; j<$alen; j++ ))
    do
        for (( i=0; i<$iteration; i++ ))
        do
             ./${app[$j]} -f $path/$dataset.bcsr -p ${parts[$k]} --length $step --source ${source[$k]} >> result/$dataset-${app[$j]}-${parts[$k]}.txt
        done
    done
done

for (( j=0; j<$alen; j++ ))
do
    for (( k=0; k<$plen; k++ ))
    do
        cat result/$dataset-${app[$j]}-${parts[$k]}.txt | grep -o "[0-9|.]* ms" | grep -o "[0-9|.]*" > result/$dataset-${app[$j]}-${parts[$k]}-t.txt
    done
done
