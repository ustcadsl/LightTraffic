#!/bin/bash

declare -a cw_walk=(74189485   111284228   148378971   222568456   296757942   445136913 593515884   890273826  1187031768  1780547653  2374063537  3561095306 4748127075  7122190612  9496254150)
cw_wlen=${#cw_walk[@]}

declare -a fs_walk=(14689053   22033579   29378106   44067159   58756212   88134319 117512425  176268638  235024851  352537276  470049702  705074553 940099405 1410149107 1880198810 2820298215 3760397621)
fs_wlen=${#fs_walk[@]}

declare -a uk_walk=(37460664    56190996    74921328   112381992   149842656   224763985 299685313   449527970   599370627   899055941  1198741255  1798111883 2397482511  3596223767  4794965023  7192447535  9589930047)
uk_wlen=${#uk_walk[@]}

for (( i=0; i<uk_wlen; i++ ));
do
    cat result-uk/pagerank-w-${uk_walk[i]}.txt | grep "[0-9|.]* ms" | grep -o "[0-9|.]*" | grep -o ".*\..*" >> uk.txt
done

for (( i=0; i<cw_wlen; i++ ));
do
    cat result-cw/pagerank-w-${cw_walk[i]}.txt | grep "[0-9|.]* ms" | grep -o "[0-9|.]*" | grep -o ".*\..*" >> cw.txt
done

for (( i=0; i<fs_wlen; i++ ));
do
    cat result-fs/pagerank-w-${fs_walk[i]}.txt | grep "[0-9|.]* ms" | grep -o "[0-9|.]*" | grep -o ".*\..*" >> fs.txt
done
