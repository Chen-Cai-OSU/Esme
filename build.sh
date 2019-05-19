#!/usr/bin/env bash
python=/Users/baidu/anaconda3/bin/python
file=/Users/baidu/Documents/Esme/Esme/permutation/replicate.py

platform='unknown'
unamestr=`uname`
echo $unamestr

if [[ "$unamestr" == 'Linux' ]]; then
    /home/cai.507/anaconda3/bin/python setup.py sdist bdist_wheel
    /home/cai.507/anaconda3/bin/pip install .
elif [[ "$unamestr" == 'Darwin' ]]; then
    sudo $python setup.py install
    sudo $python $file
fi



