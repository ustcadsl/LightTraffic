#!/bin/bash

mkdir -p result-cw

iteration=5

declare -a walk=(74189485   111284228   148378971   222568456   296757942   445136913 593515884   890273826  1187031768  1780547653  2374063537  3561095306 4748127075  7122190612  9496254150)
wlen=${#walk[@]}

path=$1
dataset=$path/clueweb.bcsr

../../pagerank -f $dataset -p 10 -w 50000000 -m -s 512

for (( i=0; i<$iteration; i++ ));
do
    for (( j=0; j<$wlen; j++ ));
    do
        ../../pagerank -f $dataset -p 2 -w ${walk[$j]} -b 80000000 >> result-cw/pagerank-w-${walk[$j]}.txt
    done
done