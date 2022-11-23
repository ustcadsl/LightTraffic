#!/bin/bash

path=/LightTraffic/dataset
make

cd /LightTraffic/benchmark/figure9 && ./run.sh $path && python3 throughput.py && python3 draw.py

cd /LightTraffic/benchmark/figure10 && ./run.sh $path && python3 draw.py

cd /LightTraffic/benchmark/figure11 && ./run.sh $path && python3 throughput.py && python3 draw.py

cd /LightTraffic/benchmark/figure12and17 && ./run.sh $path && python3 kernel-time.py && python3 reshuffle-time.py

cd /LightTraffic/benchmark/figure13 && ./run.sh $path && python3 draw.py && ./iter_analysis.sh && python3 call.py

cd /LightTraffic/benchmark/figure14 && ./run.sh $path && python3 draw.py

cd /LightTraffic/benchmark/figure15 && ./run.sh $path && python3 draw.py

cd /LightTraffic/benchmark/figure16 && ./run.sh $path && python3 draw.py

cd /LightTraffic/benchmark/figure18 && ./run.sh $path && ./collect-data.sh && python3 draw.py
