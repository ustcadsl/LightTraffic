#!/bin/bash

rm -r result
mkdir result

declare -a dataset=(livejournal orkut twitter)
declare -a source=(10009 43607 23934132)
declare -a partsize=(32 64 64)
declare -a app=(genericwalk ppr)

step=80
iteration=5


alen=${#app[@]}
dlen=${#dataset[@]}

path=$1

for (( k=0; k<$dlen; k++ ))
do
    ../../ppr -f $path/${dataset[$k]}.bcsr -w 100000 --source ${source[$k]} -m -s ${partsize[$k]}

    for (( j=0; j<$alen; j++ ))
    do
        for (( i=0; i<$iteration; i++ ))
        do
             ../../${app[$j]} -f $path/${dataset[$k]}.bcsr --length $step --source ${source[$k]} >> result/${dataset[$k]}-${app[$j]}.txt
        done
    done
done

for (( j=0; j<$alen; j++ ))
do
    for (( k=0; k<$dlen; k++ ))
    do
        cat result/${dataset[$k]}-${app[$j]}.txt | grep -o "[0-9|.]* ms" | grep -o "[0-9|.]*" > result/${dataset[$k]}-${app[$j]}-t.txt
    done
done
