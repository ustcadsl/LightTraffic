CC = nvcc -Xptxas -O3
FLAGS = -g -std=c++14
INCLUDES = -I../../AnyOption -I../../src
ANYOPTION = ../../AnyOption/anyoption.cpp

all: baseline1 baseline2 opt

baseline1: ../../app/pagerank.cu ../../src/*
	$(CC) -o baseline1 $(FLAGS) $(INCLUDES) -D ONE_LEVEL_QUEUE ../../app/pagerank.cu $(ANYOPTION)

baseline2: ../../app/pagerank.cu ../../src/*
	$(CC) -o baseline2 $(FLAGS) $(INCLUDES) -D NO_RANDOM_ACCESS_REDUCE ../../app/pagerank.cu $(ANYOPTION)

opt: ../../app/pagerank.cu ../../src/*
	$(CC) -o opt $(FLAGS) $(INCLUDES) ../../app/pagerank.cu $(ANYOPTION)

clean:
	rm opt baseline1 baseline2 
