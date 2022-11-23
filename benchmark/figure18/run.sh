#!/bin/bash

rm -rf result-*
rm cw.txt
rm uk.txt
rm fs.txt

./run-fs.sh $1
./run-uk.sh $1
./run-cw.sh $1
