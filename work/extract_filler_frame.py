#! -*- coding: utf-8 -*-
import numpy as np
import argparse
import re
import myio

def convertPmor(x):
	x = x.split('\n')
	#x[0]はファイル番号

	ret = []

	for l in x[1:]:
		st,et,t = l.split()
		if st == '-':continue

		suf = t.split(',')[0]
		# if suf.startswith('['):continue

		ret.append((st,et,suf))
	return ret

def loadPmor(fn):
	with open(fn) as fp:
		ptext = fp.read()

	pt_file = ptext.split('\n\n')
	pl = [convertPmor(x) for x in pt_file]

	return pl

def extract_silpau_time(pmor):
	silpaus = []	
	assert len(pmor)>0,pmor
	assert len(pmor[0])>0,pmor

	#最初のsil
	silpaus.append((0.0,pmor[0][0]))

	for st,et,suf in pmor:
		if suf.startswith('[IP') or suf.startswith('[P'):
			silpaus.append((st,et))

	return silpaus

def disp_silpau(silpau):
	return '\n'.join(['\t'.join([str(xx) for xx in x]) for x in silpau])

def main():

	pmors = loadPmor('./wav/hojo-disful/MHT-ATR503_reading.pmor')
	pmors.remove([])

	for i,pmor in enumerate(pmors):
		try:		
			silpau = extract_silpau_time(pmor) #sec, sections
		except:
			print(i)
			quit(i)

		silpaustr = disp_silpau(silpau)


		with open('work/silpautimes/{}.txt'.format(i+1),'w') as fp:
			fp.write(silpaustr+'\t')




if __name__=='__main__':
	main()
