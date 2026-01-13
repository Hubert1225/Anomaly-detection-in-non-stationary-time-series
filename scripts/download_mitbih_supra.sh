#!/bin/bash

mkdir ../data/raw/mitbih_supra
mkdir temp

wget https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip
unzip mit-bih-supraventricular-arrhythmia-database-1.0.0.zip -d temp/
mv temp/mit-bih-supraventricular-arrhythmia-database-1.0.0/* ../data/raw/mitbih_supra/

rm -r temp
rm mit-bih-supraventricular-arrhythmia-database-1.0.0.zip
