CC = nvcc -Xptxas -O3
FLAGS = -g -std=c++14
INCLUDES = -I../../AnyOption -I../../src
ANYOPTION = ../../AnyOption/anyoption.cpp
DEFINE = -DNO_PIPELINE -DFIFO_SCHE

all: pagerank-opt pagerank-base pagerank-asyn pagerank-fifo

pagerank-opt: ../../app/pagerank.cu ../../src/*
	$(CC) -o pagerank-opt $(FLAGS) $(INCLUDES) ../../app/pagerank.cu $(ANYOPTION)

pagerank-base: ../../app/pagerank.cu ../../src/*
	$(CC) -o pagerank-base $(FLAGS) $(INCLUDES) $(DEFINE) ../../app/pagerank.cu $(ANYOPTION)

pagerank-asyn: ../../app/pagerank.cu ../../src/*
	$(CC) -o pagerank-asyn $(FLAGS) $(INCLUDES) -D NO_PIPELINE ../../app/pagerank.cu $(ANYOPTION)

pagerank-fifo: ../../app/pagerank.cu ../../src/*
	$(CC) -o pagerank-fifo $(FLAGS) $(INCLUDES) -D FIFO_SCHE ../../app/pagerank.cu $(ANYOPTION)

clean:
	rm pagerank-opt pagerank-base pagerank-asyn pagerank-fifo