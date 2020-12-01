#! -*- coding: utf-8 -*-
import numpy as np
import argparse
import glob,myio
import subprocess

def wavsize(x):

	base,sentence = x.split()
	wavfn = 'wav/hojo-disful/wav.16k/{}.wav'.format(base)

	wavlen = subprocess.run('wavsize {}'.format(wavfn),shell=True,stdout=subprocess.PIPE,text=True)

	return int(base),sentence,float(wavlen.stdout)

def main():

	fs = ('EH','MA','ANO')

	for filler in fs:

		sl = myio.fn2list('./wav/hojo-disful/text.tag.txt')

		sl_fil = [ x for x in sl if '{}-FILLER'.format(filler) in x ]

		sl_fil = sl_fil[:10]

		sl_fil_wavlen = [wavsize(x) for x in sl_fil]

		sl_fil_wavlen = sorted(sl_fil_wavlen,key=lambda x:x[2])

		print(filler,sl_fil_wavlen[0])

if __name__=='__main__':
	main()
