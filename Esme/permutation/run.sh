#!/bin/bash
#for graph in  'imdb_binary'
for mehtod in 'deg' 'cc' 'ricciCurvature';
do
for graph in  'imdb_binary' 'dd_test' 'reddit_binary';
    do
        for  kerneltype in  'sw';
        do
        time ~/anaconda2/bin/python aux/sw.py --kerneltype=$kerneltype --graph=$graph --method=$mehtod
        done

    done
done
exit




#for graph in  'imdb_binary'
for graph in  'reddit_binary'
do
        for homtype in  '0' # '1' '01' #'0-' '0+' ;
    do
        for  dgmtype in  'normal' 'fake';
    do
        time ~/anaconda2/bin/python perm.py --no-load --comp --eval --graph=$graph --dgmtype=$dgmtype --homtype=$homtype
    done
done
done
exit
for graph in  'dd_test'
#for graph in 'mutag'
do
        for homtype in '1' '01' #'0-' '0+' ;
    do
        for  dgmtype in  'normal'  ;
    do
        time ~/anaconda2/bin/python perm.py --no-load --comp --eval --graph=$graph --dgmtype=$dgmtype --homtype=$homtype
    done
done
done