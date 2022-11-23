#!/bin/bash

make -j
mkdir -p dataset && cd dataset

wget -nv https://zenodo.org/record/7139731/files/livejournal.zip && unzip livejournal.zip && rm livejournal.zip

wget -nv https://zenodo.org/record/7139754/files/orkut.zip && unzip orkut.zip && rm orkut.zip

wget -nv https://zenodo.org/record/7139621/files/twitter.zip && unzip twitter.zip && rm twitter.zip

wget -nv https://zenodo.org/record/7134230/files/friendster.zip && unzip friendster.zip && rm friendster.zip

wget -nv https://zenodo.org/record/7131715/files/uk-union.zip && unzip uk-union.zip && rm uk-union.zip

wget -nv https://zenodo.org/record/7119052/files/yahoo.zip && unzip yahoo.zip && rm yahoo.zip

wget -nv https://nrvis.com/download/data/massive/web-ClueWeb09.zip && unzip -p web-ClueWeb09.zip > clueweb.el && rm web-ClueWeb09.zip && ../converter_txt clueweb.el && rm clueweb.el
