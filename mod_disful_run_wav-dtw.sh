#!/bin/sh

s0="./wav/clb/"
s1="./wav/bdl/"
s2="./wav/slt/"
s3="./wav/rms/"

# hojo-disful->mht
# python wav-dtw_mod.py -g -1 \
#        -i ./wav/hojo-disful/wav.16k/ ./wav/mht/numname/ \
#        -o ./out/p-0.25 \
#        -p 0.25 \
       # --debug

# mht->hojo-disful
python wav-dtw_mod.py -g -1 \
       -i ./wav/mht/numname/ ./wav/hojo-disful/wav.16k/ \
       -o ./out/mht_hojo-disful \
       -p 0.25 \

