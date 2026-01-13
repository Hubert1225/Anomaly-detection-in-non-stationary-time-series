#!/bin/bash

mkdir ../data/raw/mitbih_arrythmia
mkdir temp

wget https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
unzip mit-bih-arrhythmia-database-1.0.0.zip -d temp/
mv temp/mit-bih-arrhythmia-database-1.0.0/* ../data/raw/mitbih_arrythmia/

rm -r temp
rm mit-bih-arrhythmia-database-1.0.0.zip
