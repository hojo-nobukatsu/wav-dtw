#!/bin/sh

s0="./wav/clb/"
s1="./wav/bdl/"
s2="./wav/slt/"
s3="./wav/rms/"

python wav-dtw.py -g ${1} \
       -i ${s0} ${s3} \
       -o ./out/

