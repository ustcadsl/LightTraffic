declare -a app=(pagerank-opt pagerank-base pagerank-asyn pagerank-fifo)

declare -a parts=(150 100 50 25)

dataset=friendster


alen=${#app[@]}
plen=${#parts[@]}

for (( j=0; j<$alen; j++ ))
do
    for (( k=0; k<$plen; k++ ))
    do
        cat result/$dataset-${app[$j]}-${parts[$k]}.txt | grep -o "iterations: [0-9]*" | grep -o "[0-9]*" > result/$dataset-${app[$j]}-${parts[$k]}-calls.txt
        cat result/$dataset-${app[$j]}-${parts[$k]}.txt | grep -o "explicit: [0-9]*" | grep -o "[0-9]*" >> result/$dataset-${app[$j]}-${parts[$k]}-calls.txt
        cat result/$dataset-${app[$j]}-${parts[$k]}.txt | grep "graph loading time" | grep -o "calls: [0-9]*" | grep -o "[0-9]*" >> result/$dataset-${app[$j]}-${parts[$k]}-calls.txt
    done
done
