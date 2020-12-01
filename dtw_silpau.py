import numpy as np
from scipy.spatial.distance import cdist
from math import isinf


def dtw(x, y, dist, warp=1, w=np.inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = np.full((r + 1, c + 1), np.inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = np.zeros((r + 1, c + 1))
        D0[0, 1:] = np.inf
        D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)

    #import pdb;pdb.set_trace() # Breakpoint
    #return D1[-1, -1] / sum(D1.shape), C, D1, path
    return D1[-1, -1] / len(path[0]), C, D1, path

def dtw_endpointfree(x, y, dist, warp=1, w=np.inf, s=1.0):
    """
    Computes Endpoint-free Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = np.full((r + 1, c + 1), np.inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = np.zeros((r + 1, c + 1))
        D0[0, 1:] = np.inf
        D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)

    if np.min(D0[:,c])<np.min(D0[r,:]):
        r_end = np.argmin(D0[:,c])
        c_end = c
    else:
        r_end = r
        c_end = np.argmin(D0[r,:])
    D0 = D0[0:r_end+1,0:c_end+1]
    D1 = D1[0:r_end,0:c_end]
    
    #import pdb;pdb.set_trace() # Breakpoint
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    #return D1[-1, -1] / sum(D1.shape), C, D1, path
    return D1[-1, -1] / len(path[0]), C, D1, path

def mydtw(x,y,silpauframes, dist,w=np.inf, p=0):
    r, c = len(x), len(y)
    #r, c = D0.shape
    D0 = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            D0[i, j] = dist(x[i], y[j])
    AccDis = np.full(D0.shape, np.inf)
    AccDis[0,0] = D0[0,0]
    pointer = np.full(D0.shape, 0)
    irange = range(1,min(r,1+w+1))

    for i in irange:
        AccDis[i,0] = AccDis[i-1,0] + p + D0[i,0]
        pointer[i,0] = 1 #means "came from down"
    jrange = range(1,min(c,1+w+1))

    for j in jrange:
        AccDis[0,j] = AccDis[0,j-1] + p + D0[0,j]
        pointer[0,j] = 2 #means "came from left"

    #AccDis_without_p = AccDis
        
    for i in range(1,r):
        jrange = range(max(1,i-w),min(c,i+w+1))
        for j in jrange:
            if j in silpauframes:
                pi = -3*p #ここをパラメータにしても良い．マイナスでもいい．
            else:
                pi = p

            AccDis[i,j] = np.min([AccDis[i-1,j-1], AccDis[i-1,j]+pi, AccDis[i,j-1]+p]) + D0[i,j]
            pointer[i,j] = np.argmin([AccDis[i-1,j-1], AccDis[i-1,j]+p, AccDis[i,j-1]+p])
            #AccDis_without_p[i,j] = [AccDis[i-1,j-1], AccDis[i-1,j], AccDis[i,j-1]][pointer[i,j]] + D0[i,j]
            
    #import pdb;pdb.set_trace() # Breakpoint
    minAccDis = AccDis[r-1,c-1]
    
    # trace back
    path_r, path_c = [r-1], [c-1]
    i, j = r-1, c-1
    count = 0
    while (i > 0) or (j > 0):
        if pointer[i,j]==0:
            i -= 1
            j -= 1
        elif pointer[i,j]==1:
            i -= 1
        else: #pointer[i,j]==2:
            j -= 1
        path_r.insert(0, i)
        path_c.insert(0, j)
        count += 1

    #import pdb;pdb.set_trace() # Breakpoint
    return np.array(path_r), np.array(path_c), minAccDis/len(path_r), D0

def mydtw_endpointfree(x,y,dist, w=np.inf, p=0):
    r, c = len(x), len(y)
    #r, c = D0.shape
    D0 = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            D0[i, j] = dist(x[i], y[j])
    AccDis = np.full(D0.shape, np.inf)
    AccDis[0,0] = D0[0,0]
    pointer = np.full(D0.shape, 0)
    irange = range(1,min(r,1+w+1))
    for i in irange:
        AccDis[i,0] = AccDis[i-1,0] + p + D0[i,0]
        pointer[i,0] = 1 #means "came from down"
    jrange = range(1,min(c,1+w+1))
    for j in jrange:
        AccDis[0,j] = AccDis[0,j-1] + p + D0[0,j]
        pointer[0,j] = 2 #means "came from left"

    #AccDis_without_p = AccDis
        
    for i in range(1,r):
        jrange = range(max(1,i-w),min(c,i+w+1))
        for j in jrange:
            AccDis[i,j] = np.min([AccDis[i-1,j-1], AccDis[i-1,j]+p, AccDis[i,j-1]+p]) + D0[i,j]
            pointer[i,j] = np.argmin([AccDis[i-1,j-1], AccDis[i-1,j]+p, AccDis[i,j-1]+p])
            #AccDis_without_p[i,j] = [AccDis[i-1,j-1], AccDis[i-1,j], AccDis[i,j-1]][pointer[i,j]] + D0[i,j]
            
    if np.min(AccDis[:,c-1])<np.min(AccDis[r-1,:]):
        r_end = np.argmin(AccDis[:,c-1])
        c_end = c-1
        minAccDis = AccDis[r_end,c-1]
        #minAccDis = AccDis_without_p[r_end,c-1]
    else:
        r_end = r-1
        c_end = np.argmin(AccDis[r-1,:])
        minAccDis = AccDis[r-1,c_end]
        #minAccDis = AccDis_without_p[r-1,c_end]

    #import pdb;pdb.set_trace() # Breakpoint
        
    # trace back
    path_r, path_c = [r_end], [c_end]
    i, j = r_end, c_end
    count = 0
    while (i > 0) or (j > 0):
        if pointer[i,j]==0:
            i -= 1
            j -= 1
        elif pointer[i,j]==1:
            i -= 1
        else: #pointer[i,j]==2:
            j -= 1
        path_r.insert(0, i)
        path_c.insert(0, j)
        count += 1

    #import pdb;pdb.set_trace() # Breakpoint
    return np.array(path_r), np.array(path_c), minAccDis/len(path_r)

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def path2array(path,size):
    A = np.zeros(size)
    pathlen = len(path[0])
    for p in range(pathlen):
        A[path[0][p],path[1][p]]=1.0
    Asum=np.sum(A,axis=0,keepdims=True)
    Asum[Asum==0]=1.0
    A = A/Asum
    return A
