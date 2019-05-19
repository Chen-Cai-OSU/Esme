#!/usr/bin/env bash
for i in 1 2 3
do
    ~/anaconda3/bin/python graph/gm.py --rs $i
done