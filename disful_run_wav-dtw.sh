#!/bin/sh

s0="./wav/clb/"
s1="./wav/bdl/"
s2="./wav/slt/"
s3="./wav/rms/"

python wav-dtw.py -g -1 \
       -i ./wav/hojo-disful/wav.16k/ ./wav/mht/numname/ \
       -o ./out/p-10.0 \
       -p 10.0 \
       --debug
       # --debug