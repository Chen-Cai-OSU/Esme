#!/usr/bin/env bash

python="/home/cai.507/anaconda3/bin/python"
tudataset='/home/cai.507/Documents/DeepLearning/Esmé/Esme/graph/dataset/tu_dataset.py'
for graph in 'FRANKENSTEIN' 'BZR' 'COX2'  'DHFR'  # 'ENZYMES'
do
$python $tudataset --graph $graph
done