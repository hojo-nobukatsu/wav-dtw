#! -*- coding: utf-8 -*-
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob,pickle,os

def get_maxlen(l):

	xl = []
	yl = []

	for pf in l:
		with open(pf,'rb') as fp:
			r,c = pickle.load(fp)
		xl.append(r[-1]+1)
		yl.append(c[-1]+1)

	lx = max(xl)
	ly = max(yl)
	assert lx > ly
	return lx,ly

def plotpath(pfl,figfn):

	assert len(pfl) <= 16,len(pfl)
	cm = plt.get_cmap('jet')
	bc = cm(0) # backgroundcolor

	#batch内の最大のsrc長さ/trg長さを取得する．
	max_src_len, max_trg_len = get_maxlen(pfl)
	plt.rcParams['axes.facecolor'] = bc

	# tt_src = np.arange(1.0*max_src_len) * 

	for i, pf in enumerate(pfl):
		base = os.path.basename(pf).split('.')[0].split('-')[0]
		col = i % 4
		row = i //4
		ax = plt.subplot(4,4,i+1)

		plt.xticks(color='None')
		plt.yticks(color='None')		

		with open(pf,'rb') as fp:
			p_r,p_c = pickle.load(fp)

		assert len(p_r) == len(p_c)
		Lr = p_r[-1] + 1
		Lc = p_c[-1] + 1

		# plt.pcolormesh(Amat.T,rasterized = True)
		plt.plot(p_r,p_c,'y')

		plt.xlim((0,max_src_len))
		plt.ylim((0,max_trg_len))
		plt.title(base,fontsize=10)
	plt.savefig(figfn)
	plt.clf()

def main():

	pathpath = 'out/mht_hojo-disful_no0dim/'
	pl = sorted(glob.glob(pathpath+'/*_path.pickle'))

	pl_train = pl[:478]
	pl_test = pl[:478:]

	print(pl_train)
	print(pl_test)

	print(len(pl_train))
	print(len(pl_test))	

	N = len(pl_train) # 478
	print(N)
	bs = 16

	for i,n in enumerate(range(0,N,16)):
		pfl = pl_train[n:n+bs]

		figfn = 'out/pathplot.mht_hojo-disful_no0dim/batch{}.png'.format(str(i).zfill(4))

		plotpath(pfl,figfn)
		print(figfn)

if __name__=='__main__':
	main()
