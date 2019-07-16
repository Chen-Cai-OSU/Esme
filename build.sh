#!/usr/bin/env bash
python=~/anaconda3/bin/python
pip=~/anaconda3/bin/pip

$python setup.py sdist bdist_wheel
$pip install .

#/home/cai.507/anaconda2/bin/python setup.py sdist bdist_wheel
#/home/cai.507/anaconda2/bin/pip install .