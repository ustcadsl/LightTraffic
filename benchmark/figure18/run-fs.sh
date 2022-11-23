#!/bin/bash

mkdir -p result-fs

iteration=5

declare -a walk=(14689053   22033579   29378106   44067159   58756212   88134319 117512425  176268638  235024851  352537276  470049702  705074553 940099405 1410149107 1880198810 2820298215 3760397621)
wlen=${#walk[@]}

path=$1
dataset=$path/friendster.bcsr

../../pagerank -f $dataset -w 50000000 -m -s 512

for (( i=0; i<$iteration; i++ ));
do
    for (( j=0; j<$wlen; j++ ));
    do
        ../../pagerank -f $dataset -p 2 -w ${walk[$j]} -b 80000000 >> result-fs/pagerank-w-${walk[$j]}.txt
    done
done
