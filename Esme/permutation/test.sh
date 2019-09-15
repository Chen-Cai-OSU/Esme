#!/usr/bin/env bash
# large scale test
python="/home/cai.507/anaconda3/bin/python"
file="/home/cai.507/Documents/DeepLearning/Esmé/Esme/permutation/config/util.py"

for feat in   'pervec'   # 'pss' 'pf' 'wg' # 'pf' 'pss' 'wg' #
do
for graph in  'protein_data' 'bzr' 'cox2' 'dhfr' 'dd_test' 'frankenstein' 'imdb_binary' 'imdb_multi' 'nci1' 'reddit_binary' #'bzr' 'imdb_binary' #'bzr' 'cox2' 'dfhr' 'dd_test' 'nci1'  'frankenstein' 'protein_data'   'imdb_binary'  'imdb_multi' 'reddit_binary' 'reddit_5K' # reddit_binary ricci has some problem #'syn1' #'reddit_binary' 'nci109' #'reddit_5K'     #'dd_test' 'nci1' #'nci109' 'protein_data' 'imdb_binary' #'imdb_multi' 'collab' 'reddit_5K' #'imdb_multi' #'nci1' 'reddit_5K' #'reddit_5K' 'reddit_12K' 'collab'
do
for fil in 'fiedler_s' # 'hks_100'   #'fiedler_s' # 'hks_100' # 'fiedler' # 'random' 'hks_1' 'hks_0.1' 'hks_10'
do
for epd in False  True
do
for flip in False
do
for permute in False  True
do
for n_cv in 1 # 10
do
#time $python $file --feat $feat --graph $graph --fil $fil --epd $epd --flip $flip --permute $permute --ntda False --n_cv $n_cv     # original
time $python $file --feat $feat --graph $graph --fil $fil --epd $epd --flip $flip --permute $permute --ntda True  --n_cv $n_cv      # turn off tda
done
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