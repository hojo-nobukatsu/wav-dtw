#! -*- coding: utf-8 -*-
import numpy as np
import argparse
import itertools
import os
import subprocess

def spk2dir(spk):
	if spk in ('hojo-disful',):
		spkdir = 'wav/{}/wav.16k/'.format(spk)
	elif spk in ('mht'):
		spkdir = 'wav/{}/numname/'.format(spk)
	else:
		spkdir = 'wav/{}/'.format(spk)

	return spkdir

def main():
	spks = ['hojo-disful','mht','mho','mmy','msh','mtk','myi']
	nspk = len(spks)

	for i,j in itertools.product(range(nspk),range(nspk)):
		if i==j:continue

		src = spks[i]
		trg = spks[j]

		srcdir = spk2dir(src)
		trgdir = spk2dir(trg)

		outdir = 'out/multispk/{src}_{trg}/'.format(src=src,trg=trg)

		if not os.path.exists(outdir):
			os.mkdir(outdir)

		com = 'python wav-dtw_mod.py -g -1 -i {srcdir} {trgdir} -o {outdir} -p 0.25 --debug'.format(srcdir=srcdir,trgdir=trgdir,outdir=outdir)

		proc = subprocess.run(com,shell=True,text=True)
		print(proc)

if __name__=='__main__':
	main()
