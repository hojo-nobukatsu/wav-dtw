import os
import numpy as np
import math, argparse, random
import sys
import scipy
from scipy import signal
from scipy.io import wavfile
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import six
import pysptk
from pysptk import sptk
# import pymreps
# from pymreps import pymreps
import pyworld as pw
#from pyspace import pyspace
from dtw import dtw
from dtw import dtw_endpointfree
from dtw_mod import mydtw_endpointfree, mydtw
from dtw import path2array
from datetime import datetime
import pickle

#Usage: See run_mgcep-dtw.sh

def load_wav(path):
    fs, _x = wavfile.read(path)
    if _x.ndim==2:
        x = np.mean(_x, axis=0)
    else:
        x =_x
    x = x.flatten()
    return fs, x

def get_mcep_condition(fs, dim=None):
    if int(fs) == 8000:
        if dim is None: dim = 23
        alpha = 0.31
    elif int(fs) == 10000:
        if dim is None: dim = 23
        alpha = 0.35
    elif int(fs) == 12000:
        if dim is None: dim = 23
        alpha = 0.37
    elif int(fs) == 16000:
        if dim is None: dim = 27
        alpha = 0.41
    elif int(fs) == 20000:
        if dim is None: dim = 31
        alpha = 0.44
    elif int(fs) == 22050:
        if dim is None: dim = 35
        alpha = 0.455
    elif int(fs) == 44100:
        if dim is None: dim = 43
        alpha = 0.544
    elif int(fs) == 48000:
        if dim is None: dim = 43
        alpha = 0.554
    else:
        print("Undefined sampling rate")
        assert False
    return dim, alpha
        
def refine(sig, fs, f0, shiftms):
    _f0, t = pw.dio(sig, fs, frame_period=shiftms)
    length = min(len(f0), len(_f0))
    _f0[:length] = f0[:length]
    return _f0, t

def interp(f0,shiftms):
    cf0, vad = pyspace.refine(f0, vad=None, nframe=0, shiftms=shiftms, logscale=False,
                              interp=True, smooth=True, pre_remove_ms=50, highcut=10.0, butord=5)
    return cf0, vad

def lengthequalize(f0,wsp,wmc):
    framenum_f0=len(f0)
    framenum_wsp = wsp.shape[0]
    framenum_wmc = wmc.shape[0]
    minlen = min(framenum_f0,framenum_wsp,framenum_wmc)
    _f0 = f0[0:minlen]
    _wsp = wsp[0:minlen,:]
    _wmc = wmc[0:minlen,:]
    return _f0, _wsp, _wmc

def wav2wmc(wpath, flen, fshift_ms, melcep='codespec', fzero='harvest', mg_order=None, mg_alpha=None):

    #print(wpath)
    fs, y = load_wav(wpath)
    y = y.astype(np.float64)

    flen_ms = flen/fs*1000
    fshift = int(fshift_ms/1000*fs)
        
    nsamples = len(y)
    #print('Sampling rate: ', fs)
    #print('Sample num: ', nsamples)
    #print('Frame length: {}(ms)'.format(flen_ms))
    #print('Frame shift: {}(ms)'.format(fshift_ms))
    
    print('F0 extractor: {}'.format(fzero))
    # mREPS pitch extractor
    # f0_pymreps = pymreps.analysis(y, fs=fs, maxf0=500.0, minf0=50.0,
                                  # shiftms=fshift_ms, fftp=512, get_uv=False)
    #import pdb;pdb.set_trace() # Breakpoint
    # f0_pymreps, t = refine(y, fs, f0_pymreps, fshift_ms)
    #f0_pymreps,vad = interp(f0_pymreps,fshift_ms)
    if fzero == 'pymreps':
        f0 = f0_pymreps
    if fzero == 'harvest':
        _f0, t = pw.harvest(y, fs, frame_period=fshift_ms)
        f0 = pw.stonemask(y, _f0, t, fs)
    if fzero == 'dio':
        _f0, t = pw.dio(y, fs, frame_period=fshift_ms)
        f0 = pw.stonemask(y, _f0, t, fs)

    # WORLD spectral envelope extraction
    wsp = pw.cheaptrick(y, f0, t, fs)
    ap = pw.d4c(y, f0, t, fs)

    print('Mel-cepstrum analysis: {}'.format(melcep))
    # Mel-generalized cepstral analysis
    if mg_order is None:
        mg_order = get_mcep_condition(fs)[0]
    if mg_alpha is None:
        mg_alpha = get_mcep_condition(fs)[1]
    if melcep == 'pysptk':
        wmc = pysptk.sp2mc(wsp, mg_order, mg_alpha)
    if melcep == 'codespec':
        wmc = pw.code_spectral_envelope(wsp, fs, mg_order+1)

    #import pdb;pdb.set_trace() # Breakpoint 
        
    #Exclude 0-th order
    #wmc = wmc[:,1:]

    #f0,wsp,wmc=lengthequalize(f0,wsp,wmc)
    #cf0,vad = interp(f0,fshift_ms)
    #Convert linear F0 contour into log F0 contour
    #lcf0 = np.zeros(len(cf0))
    #lcf0[cf0!=0] = np.log(cf0[cf0!=0])
    
    return wmc, f0, ap, fs

def wmc2wav(wmc, f0, ap, flen, fshift_ms, fs,
            melcep='codespec',fzero='harvest',mg_order=None,mg_alpha=None):
    # Synthesize waveform
    wmc = wmc.copy(order='C')
    if mg_order is None:
        mg_order = get_mcep_condition(fs)[0]
    if mg_alpha is None:
        mg_alpha = get_mcep_condition(fs)[1]
    if melcep == 'pysptk':
        wpsp = pysptk.mc2sp(wmc, mg_alpha, flen)
    if melcep == 'codespec':
        wpsp = pw.decode_spectral_envelope(wmc, fs, flen)

    #import pdb;pdb.set_trace() # Breakpoint 
    x = pw.synthesize(f0, wpsp, ap, fs, frame_period=fshift_ms)
    x = x/max(abs(x))*30000

    return x

def shorten_path(pr,pc):
    """
    もしpcの中に重複indexがあれば，prとpcからそれを削除する
    """

    prs = []
    pcs = []

    ic_b = -1

    for ir, ic in zip(pr,pc):
        if ic == ic_b:
            #重複
            continue
        else:
            #重複なし
            prs.append(ir)
            pcs.append(ic)
            ic_b = ic

    return prs,pcs

def vectorize_path(r,c):
    """
    rの次元にa合わせてcの値をベクトルで表現して返す
    """

    # assert r[-1] > c[-1],'{},{}'.format(r[-1],c[-1])
    if r[-1] < c[-1]:
        print('WARNING: r is longer (r:{}),(c:{})).'.format(r[-1],c[-1]))

    N = r[-1]

    v = np.zeros((N,),dtype=np.float32)

    for n in range(N):
        ids = np.where(r==n)[0]
        v[n] = np.average(c[ids])

    return v

def main():
    parser = argparse.ArgumentParser(description='Generate time-aligned stereo WAV file from a pair of WAV inputs')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument(
        '--inpath', '-i', type=str, nargs='+', action='store', required=True, help='directories of the wav data to be compared')
    parser.add_argument(
        '--flen', '-l', type=int, default=1024, help='Frame length')
    parser.add_argument(
        '--fshift_ms', '-s', type=float, default=8.0, help='Frame shift in ms')
    parser.add_argument(
        '--melcep', '-mcep', default='codespec', type=str, help='mel-cepstrum extractor: codespec, pysptk')
    parser.add_argument(
        '--fzero', '-fz', default='harvest', type=str, help='F0 extractor: pymreps, harvest, dio')
    parser.add_argument(
        '--mg_order', '-q', type=int, default=None, help='Order of mel-generalized cepstral representation')
    parser.add_argument(
        '--mg_alpha', '-m', type=float, default=None, help='aplha parameter of mel-generalized cepstral representation')
    parser.add_argument(
        '--diagonal-penalty', '-p', type=float, default=0.25, help='penalty for path not going diagonal')
    parser.add_argument(
        '--savepath', '-o', type=str, default='./out/', help='directory of the txt files of the results')
    parser.add_argument(
        '--debug', action='store_true', default=False, help='if True, use only 10 files.')
    parser.add_argument(
        '--use-dim0', action='store_true', default=False, help='use 0-th cepstrum to calculate distance matrices')    
    args = parser.parse_args()

    flen = args.flen
    fshift_ms = args.fshift_ms
    inpath = args.inpath
    assert len(inpath)==2
    melcep = args.melcep
    fzero = args.fzero
    mg_order = args.mg_order
    mg_alpha = args.mg_alpha
    shorten = False
    penalty = args.diagonal_penalty
    print(penalty)

    # Extract the list of wav file names only
    wavFileNames=[]
    for i in range(0,2):
        wavFileNames.append([])
        tmp = sorted(os.listdir(inpath[i]))
        for f in range(len(tmp)):
            if os.path.splitext(tmp[f])[1]=='.wav':
                wavFileNames[i].append(tmp[f])

    if args.debug:
        wavFileNames[0] = wavFileNames[0][:2]
        wavFileNames[1] = wavFileNames[1][:2]

    #import pdb;pdb.set_trace() # Breakpoint
                
    #wavFileNames=[]
    #wavFileNames.append(sorted(os.listdir(inpath[0])))
    #wavFileNames.append(sorted(os.listdir(inpath[1])))
    #assert len(wavFileNames[0])==len(wavFileNames[1])
    wavFileNum = len(wavFileNames[0])
    savepath = args.savepath

    # Make directories
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    mydis = lambda x, y: np.linalg.norm(x-y, ord=2)

    # wavFileNum = min(wavFileNum,30)

    #accdis = np.zeros(wavFileNum)
    #lf0corr = np.zeros(wavFileNum)
    for f in range(wavFileNum):
        #import pdb;pdb.set_trace() # Breakpoint
        # File name extraction
        fname_head1 = os.path.splitext(wavFileNames[0][f])[0]
        fname_head2 = os.path.splitext(wavFileNames[1][f])[0]

        # Feature extraction
        wmc1, f01, ap1, fs1 = wav2wmc(os.path.join(inpath[0],wavFileNames[0][f]),
                                      flen,fshift_ms,melcep,fzero,mg_order,mg_alpha)
        wmc2, f02, ap2, fs2 = wav2wmc(os.path.join(inpath[1],wavFileNames[1][f]),
                                      flen,fshift_ms,melcep,fzero,mg_order,mg_alpha)
        gm1 = np.mean(wmc1,axis=0,keepdims=True)
        gs1 = np.std(wmc1,axis=0,keepdims=True)
        gm2 = np.mean(wmc2,axis=0,keepdims=True)
        gs2 = np.std(wmc2,axis=0,keepdims=True)

        #wmc1 = (wmc1-gm1)/gs1
        #wmc2 = (wmc2-gm2)/gs2        
        wmc1_conv = (wmc1-gm1)/gs1*gs2+gm2
        wmc1_conv = np.maximum(wmc1_conv,np.min(wmc2))
        wmc1_conv = np.minimum(wmc1_conv,np.max(wmc2))
        #import pdb;pdb.set_trace() # Breakpoint

        # DTW
        # path_r, path_c, tmpdis = mydtw_endpointfree(wmc1_conv[:,:], wmc2[:,:], dist=mydis, w=np.inf, p=0.25)
        if args.use_dim0:
        	path_r, path_c, tmpdis,D = mydtw(wmc1_conv[:,:], wmc2[:,:], dist=mydis, w=np.inf, p=penalty)
        else:
        	path_r, path_c, tmpdis,D = mydtw(wmc1_conv[:,1:], wmc2[:,1:], dist=mydis, w=np.inf, p=penalty)


        if shorten:
            path_r, path_c = shorten_path(path_r,path_c)

        path_v = vectorize_path(path_r,path_c)

        wmc1_warp = wmc1[path_r,:]
        f01_warp = f01[path_r]
        ap1_warp = ap1[path_r,:]
        
        wmc2_warp = wmc2[path_c,:]
        f02_warp = f02[path_c]
        ap2_warp = ap2[path_c,:]

        #wmc1 = wmc1*gs1+gm1
        #wmc2 = wmc2*gs2+gm2
        #wmc1_warp = wmc1_warp*gs1+gm1
        #wmc2_warp = wmc2_warp*gs2+gm2
        
        x1 = wmc2wav(wmc1, f01, ap1, flen, fshift_ms, fs1, melcep, fzero, mg_order, mg_alpha)
        x2 = wmc2wav(wmc2, f02, ap2, flen, fshift_ms, fs1, melcep, fzero, mg_order, mg_alpha)
        #import pdb;pdb.set_trace() # Breakpoint
        maxsamplenum = max(len(x1),len(x2))
        x = np.zeros((maxsamplenum,2))
        x[0:len(x1),0] = x1
        x[0:len(x2),1] = x2
        #x = np.concatenate([x1[:,np.newaxis],x2[:,np.newaxis]],1)
        y1 = wmc2wav(wmc1_warp, f01_warp, ap1_warp, flen, fshift_ms, fs1, melcep, fzero, mg_order, mg_alpha)
        y2 = wmc2wav(wmc2_warp, f02_warp, ap2_warp, flen, fshift_ms, fs1, melcep, fzero, mg_order, mg_alpha)
        maxsamplenum = max(len(y1),len(y2))
        y = np.zeros((maxsamplenum,2))
        y[0:len(y1),0] = y1
        y[0:len(y2),1] = y2
        #y = np.concatenate([y1[:,np.newaxis],y2[:,np.newaxis]],1)

        outpath = os.path.join(savepath,'{}-{}_original.wav'.format(fname_head1,fname_head2))
        wavfile.write(outpath, fs1, x.astype(np.int16))

        if shorten:
            outpath = os.path.join(savepath,'{}-{}_warpedshorten.wav'.format(fname_head1,fname_head2))            
        else:
            outpath = os.path.join(savepath,'{}-{}_warped.wav'.format(fname_head1,fname_head2))
        wavfile.write(outpath, fs1, y.astype(np.int16))

        pathpath = '{}/{}-{}_path.pickle'.format(savepath,fname_head1,fname_head2)
        with open(pathpath,'wb') as fp:
            pickle.dump((path_r,path_c),fp)
        print(pathpath)

        vecpath = '{}/{}-{}_vpath.pickle'.format(savepath,fname_head1,fname_head2)
        with open(vecpath,'wb') as fp:
            pickle.dump(path_v,fp)
        print(vecpath)
            
if __name__ == '__main__':
    main()


