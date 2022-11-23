#!/bin/bash

rm -r result
mkdir -p result
make -j

declare -a dataset=(friendster uk-union yahoo clueweb)
declare -a source=(3142233 57212565 115384435 0)
declare -a parts=(9999 120 5 13)
declare -a batch=(9900000000 9900000000 700000000 1684868322)
declare -a partsize=(64 128 3072 256)
declare -a app=(pagerank genericwalk ppr)

step=80
iteration=5


alen=${#app[@]}
dlen=${#dataset[@]}

path=$1

for (( k=0; k<$dlen; k++ ))
do
    ./ppr -f $path/${dataset[$k]}.bcsr -p ${parts[$k]} -w 100000000 --source ${source[$k]} -m -s ${partsize[$k]}

    for (( j=0; j<$alen; j++ ))
    do
        for (( i=0; i<$iteration; i++ ))
        do
             ./${app[$j]} -f $path/${dataset[$k]}.bcsr -p ${parts[$k]} -b ${batch[$k]} --length $step --source ${source[$k]} >> result/${dataset[$k]}-${app[$j]}.txt
        done
    done
done

for (( j=0; j<$alen; j++ ))
do
    for (( k=0; k<$dlen; k++ ))
    do
        cat result/${dataset[$k]}-${app[$j]}.txt | grep "[0-9|.]* ms" | grep -o "[0-9|.]*" | grep -o ".*\..*" > result/${dataset[$k]}-${app[$j]}-t.txt
    done
done
