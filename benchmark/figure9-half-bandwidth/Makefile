CC = nvcc -Xptxas -O3
FLAGS = -g -std=c++14
INCLUDES = -I../../AnyOption -I../../src
ANYOPTION = ../../AnyOption/anyoption.cpp
DEFINE = -DHALF_PCIE_BANDWIDTH -DNO_EVENT_TIMER

all: pagerank ppr genericwalk

pagerank: ../../app/pagerank.cu ../../src/*
	$(CC) -o pagerank $(FLAGS) $(INCLUDES) $(DEFINE) ../../app/pagerank.cu $(ANYOPTION)

ppr: ../../app/ppr.cu ../../src/*
	$(CC) -o ppr $(FLAGS) $(INCLUDES) $(DEFINE) ../../app/ppr.cu $(ANYOPTION)

genericwalk: ../../app/genericwalk.cu ../../src/*
	$(CC) -o genericwalk $(FLAGS) $(INCLUDES) $(DEFINE) ../../app/genericwalk.cu $(ANYOPTION)

clean:
	rm pagerank ppr genericwalk
