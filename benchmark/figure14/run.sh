#!/bin/bash

iteration=5

make -j
rm -r result
mkdir -p result

declare -a dataset=(uk-union yahoo clueweb)
declare -a source=(57212565 115384435 0)
declare -a parts=(120 5 13)
declare -a batch=(9900000000 700000000 1684868322)
declare -a partsize=(128 3072 256)

declare -a method=(opt-ppr disable-ppr zerocopy-ppr opt-pagerank disable-pagerank zerocopy-pagerank)

dlen=${#dataset[@]}
mlen=${#method[@]}

path=$1

for (( j=0; j<$mlen; j++ ))
do
    for (( k=0; k<$dlen; k++ ))
    do
        ./${method[$j]} -f $path/${dataset[$k]}.bcsr -p ${parts[$k]} --source ${source[$k]} -b ${batch[$k]} --length 80 -m -s ${partsize[$k]} >> result/${dataset[$k]}-${method[$j]}.txt

        for (( i=1; i<$iteration; i++ ))
        do
            ./${method[$j]} -f $path/${dataset[$k]}.bcsr -p ${parts[$k]} --source ${source[$k]} -b ${batch[$k]} --length 80 >> result/${dataset[$k]}-${method[$j]}.txt
        done
    done
done

for (( j=0; j<$mlen; j++ ))
do
    for (( k=0; k<$dlen; k++ ))
    do
        cat result/${dataset[$k]}-${method[$j]}.txt | grep -o "[0-9|.]* ms" | grep -o "[0-9|.]*" > result/${dataset[$k]}-${method[$j]}-t.txt
    done
done
