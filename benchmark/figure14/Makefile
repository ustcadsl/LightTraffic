CC = nvcc -Xptxas -O3
FLAGS = -g -std=c++14
INCLUDES = -I../../AnyOption -I../../src
ANYOPTION = ../../AnyOption/anyoption.cpp

all: opt-ppr disable-ppr zerocopy-ppr opt-pagerank disable-pagerank zerocopy-pagerank

opt-ppr: ../../app/ppr.cu ../../src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) $< $(ANYOPTION)

disable-ppr: ../../app/ppr.cu ../../src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) -DD_ZEROCOPY_THRESHOLD=1 $< $(ANYOPTION)

zerocopy-ppr: ../../app/ppr.cu ../../src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) -DD_ZEROCOPY_THRESHOLD_INF $< $(ANYOPTION)

opt-pagerank: ../../app/pagerank.cu ../../src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) $< $(ANYOPTION)

disable-pagerank: ../../app/pagerank.cu ../../src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) -DD_ZEROCOPY_THRESHOLD=1 $< $(ANYOPTION)

zerocopy-pagerank: ../../app/pagerank.cu ../../src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) -DD_ZEROCOPY_THRESHOLD_INF $< $(ANYOPTION)

clean:
	rm opt-ppr disable-ppr zerocopy-ppr opt-pagerank disable-pagerank zerocopy-pagerank
