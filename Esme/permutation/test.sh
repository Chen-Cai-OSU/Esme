#!/usr/bin/env bash
python="/home/cai.507/anaconda3/bin/python"
file="/home/cai.507/Documents/DeepLearning/Esmé/Esme/permutation/config/util.py"
#$python $file print_config with kwargs.bw=0.7

for feat in 'pss' # 'sw'
do
for graph in 'nci1' #'imdb_multi' 'nci1' 'reddit_5K' #'reddit_5K' 'reddit_12K' 'collab'
do
for fil in 'deg' 'ricci' 'cc' 'random' 'fiedler'
do
for epd in True False
do
for flip in True False
do
for permute in False True
do
time $python $file --feat $feat --graph $graph --fil $fil --epd $epd --flip $flip --permute $permute

done
done
done
done
done
done


exit
for fil in 'deg' #'cc' 'random' 'fiedler' 'ricci'
do
for graph in 'mutag'  #'reddit_12K' 'collab' #'reddit_5K' #'nci1' 'reddit_binary' 'reddit_5K'  # 'imdb_binary' 'imdb_multi' 'mutag' 'ptc'
do

# wg
for bw in 1 10 0.1
do
for p in 1 10
do
for K in 1
do
time $python $file with graph=$graph fil=$fil permute=False n_cv=1 feat_kwargs.bw=$bw feat='wg' feat_kwargs.K=$K feat_kwargs.p=$p
done
done
done

# sw
for bw in 1 10 0.1
do
time $python $file with graph=$graph fil=$fil permute=False n_cv=1 feat_kwargs.bw=$bw feat='sw' feat_kwargs.n_d=10
done
continue
# pi
time $python $file with graph=$graph fil=$fil permute=False n_cv=1 feat='pi'

# pss
for bw in 1 10 0.1
do
time $python $file with graph=$graph fil=$fil permute=False n_cv=1 feat_kwargs.bw=$bw feat='pss'
done



done
done