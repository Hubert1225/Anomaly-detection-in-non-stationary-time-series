#!/bin/bash

mkdir ../data/raw/ucr
mkdir temp

wget https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
unzip UCR_TimeSeriesAnomalyDatasets2021.zip -d temp
mv temp/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/* ../data/raw/ucr/

rm -r temp/
rm UCR_TimeSeriesAnomalyDatasets2021.zip
