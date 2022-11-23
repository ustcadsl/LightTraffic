CC = nvcc -Xptxas -O3
FLAGS = -g -std=c++14
INCLUDES = -I../../AnyOption -I../../src
ANYOPTION = ../../AnyOption/anyoption.cpp
DEFINE = -DNO_EVENT_TIMER

all: pagerank ppr genericwalk

pagerank: ../../app/pagerank.cu ../../src/*
	$(CC) -o pagerank $(FLAGS) $(DEFINE) $(INCLUDES) ../../app/pagerank.cu $(ANYOPTION)

ppr: ../../app/ppr.cu ../../src/*
	$(CC) -o ppr $(FLAGS) $(DEFINE) $(INCLUDES) ../../app/ppr.cu $(ANYOPTION)

genericwalk: ../../app/genericwalk.cu ../../src/*
	$(CC) -o genericwalk $(FLAGS) $(DEFINE) $(INCLUDES) ../../app/genericwalk.cu $(ANYOPTION)

clean:
	rm pagerank ppr genericwalk