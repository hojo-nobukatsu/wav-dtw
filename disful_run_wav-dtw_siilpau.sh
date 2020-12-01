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
python wav-dtw_silpau.py -g -1 \
       -i ./wav/hojo-disful/wav.16k/ ./wav/mht/numname/ \
       -o ./out/mht_hojo-disful_silpau \
       -p 0.25 \
       --silpau-path ./work/silpautimes