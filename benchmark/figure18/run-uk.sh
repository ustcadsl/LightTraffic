#!/bin/bash

mkdir -p result-uk

iteration=5

declare -a walk=(37460664    56190996    74921328   112381992   149842656   224763985 299685313   449527970   599370627   899055941  1198741255  1798111883 2397482511  3596223767  4794965023  7192447535  9589930047)
wlen=${#walk[@]}

path=$1
dataset=$path/uk-union.bcsr

../../pagerank -f $dataset -p 10 -w 50000000 -m -s 512

for (( i=0; i<$iteration; i++ ));
do
    for (( j=0; j<$wlen; j++ ));
    do
        ../../pagerank -f $dataset -p 2 -w ${walk[$j]} -b 80000000 >> result-uk/pagerank-w-${walk[$j]}.txt
    done
done