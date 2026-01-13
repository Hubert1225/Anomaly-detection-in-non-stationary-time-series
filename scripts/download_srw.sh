#!/bin/bash

mkdir ../data/raw/srw
mkdir temp

wget https://helios2.mi.parisdescartes.fr/~themisp/norma/data/Synthetic_datasets.zip
unzip Synthetic_datasets.zip -d temp/
mv temp/Synthetic_Sin+anom/noNoise_varLenghtAnomalies/* ../data/raw/srw/
mv temp/Synthetic_Sin+anom/noNoise_varNumberAnomalies/* ../data/raw/srw/
mv temp/Synthetic_Sin+anom/varNoise_fixedNumberAnomaliesfixedLength/* ../data/raw/srw/

rm -r temp/
rm Synthetic_datasets.zip
