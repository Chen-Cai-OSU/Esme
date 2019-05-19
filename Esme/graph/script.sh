#!/usr/bin/env bash

for dataset in 'reddit_binary' #'reddit_5K' #'imdb_binary' 'imdb_multi' 'collab' #
do
for fil_method in 'combined' #'node' 'edge' 'combined'
do
    time ~/anaconda3/bin/python -W ignore graph/collabration.py --fil_method $fil_method --dataset $dataset &
done
wait
done


exit
for q in 0.1 0.2 0.3 0.35 0.4 0.5 #0.1 0.2 0.3 0.4 0.5
 do
 for fil_method in 'edge' #'node' 'edge' 'combined'
    do
        time ~/anaconda3/bin/python -W ignore graph/2sbm_gc.py --q $q --fil_method $fil_method &
    done
 done


exit
for p in  0.2 0.3
do
#for zigzag in Falses
#do
    ~/anaconda3/bin/python -W ignore graph/edge_smoothing.py --p $p
#for fil in 'lap' #'random' 'deg'
#do
#~/anaconda3/bin/python -W ignore graph/gm.py --rs 1 --p $p --fil $fil
#done
done
