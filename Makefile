CC = nvcc -Xptxas -O3
FLAGS = -g -std=c++14
INCLUDES = -IAnyOption -Isrc
ANYOPTION = AnyOption/anyoption.cpp

.PHONY: clean zip build-docker run-docker dataset

all: pagerank ppr genericwalk converter_trw converter_txt rm_isolated_vertices

%: app/%.cu src/*
	$(CC) -o $@ $(FLAGS) $(INCLUDES) $< $(ANYOPTION)

converter_trw: tools/convert_from_ThunderRW.cc
	g++ -O3 -o $@ $(FLAGS) $<

converter_txt: tools/convert_from_txt.cc
	g++ -O3 -o $@ $(FLAGS) $<

rm_isolated_vertices: tools/rm_isolated_vertices.cc
	g++ -O3 -o $@ $(FLAGS) $<

clean:
	rm pagerank ppr genericwalk converter_trw converter_txt rm_isolated_vertices

zip:
	mv benchmark/*/*.pdf . && zip figures.zip *.pdf

build-docker:
	sudo docker build -t light_traffic .

run-docker: build-docker
	sudo docker run --rm -v "$(shell pwd)":/LightTraffic -it --gpus all light_traffic

dataset: dataset/livejournal.bcsr dataset/orkut.bcsr dataset/twitter.bcsr dataset/friendster.bcsr dataset/uk-union.bcsr dataset/yahoo.bcsr dataset/clueweb.bcsr
CD_DATASET = mkdir -p dataset && cd dataset

dataset/livejournal.bcsr:
	$(CD_DATASET) && wget -nv https://zenodo.org/record/7139731/files/livejournal.zip && unzip livejournal.zip && rm livejournal.zip

dataset/orkut.bcsr:
	$(CD_DATASET) && wget -nv https://zenodo.org/record/7139754/files/orkut.zip && unzip orkut.zip && rm orkut.zip

dataset/twitter.bcsr:
	$(CD_DATASET) && wget -nv https://zenodo.org/record/7139621/files/twitter.zip && unzip twitter.zip && rm twitter.zip

dataset/friendster.bcsr:
	$(CD_DATASET) && wget -nv https://zenodo.org/record/7134230/files/friendster.zip && unzip friendster.zip && rm friendster.zip

dataset/uk-union.bcsr:
	$(CD_DATASET) && wget -nv https://zenodo.org/record/7131715/files/uk-union.zip && unzip uk-union.zip && rm uk-union.zip

dataset/yahoo.bcsr:
	$(CD_DATASET) && wget -nv https://zenodo.org/record/7119052/files/yahoo.zip && unzip yahoo.zip && rm yahoo.zip

dataset/clueweb.bcsr: converter_txt
	$(CD_DATASET) && wget -nv https://nrvis.com/download/data/massive/web-ClueWeb09.zip && unzip -p web-ClueWeb09.zip > clueweb.el && rm web-ClueWeb09.zip && ../converter_txt clueweb.el && rm clueweb.el
