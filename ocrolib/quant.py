import code,pickle,sys,os,re
from pylab import *
from optparse import OptionParser
from scipy import stats
import common as ocrolib
from native import *
from numpy import *

verbose = 0

def rchoose(k,n):
    assert k<=n
    return random.permutation(range(n))[:k]
def rowwise(f,data,samples=None):
    assert data.ndim==2
    if samples is None: samples = range(len(data))
    return array([f(data[i]) for i in samples])
def argmindist_slow(x,data):
    dists = [distsq(x,v) for v in data]
    return argmin(dists)
def dist(u,v):
    return linalg.norm(u-v)
def distsq(x,y):
    d = x-y
    return dot(d.ravel(),d.ravel())

nmod = compile_and_load(r'''
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void alldists(int r,int d,float out[r],float v[d],float vs[r][d]) {
#pragma omp parallel for
    for(int i=0;i<r;i++) {
        double total = 0.0;
        for(int j=0;j<d;j++) {
            float delta = v[j]-vs[i][j];
            total += delta*delta;
        }
        assert(!isnan(total));
        out[i] = total; 
    }
}

int findeps(int r,int d,float v[d],float vs[r][d],int eps) {
}
''')

nmod.alldists.argtypes = [I,I,A1F,A1F,A2F]

def alldists(v,data,out=None):
    if out is None: out = zeros(data.shape[0],'f')
    assert len(out)==data.shape[0]
    assert len(v)==data.shape[1]
    nmod.alldists(out,v,data)

def argmindist2(v,data):
    assert len(v)==data.shape[1]
    ds = zeros(data.shape[0],'f')
    nmod.alldists(data.shape[0],data.shape[1],ds,v,data)
    i = argmin(ds)
    return i,ds[i]

def kmeans(data,k,maxiter=100,minchange=1,outlier=3.0,minvecs=5):
    """Regular k-means algorithm.  Computes k means from data."""
    assert data.dtype==numpy.dtype('f')
    global verbose, CHECK
    n = len(data)
    d = len(data[0])
    means = data[rchoose(k,n)]
    oldmins = -ones(n,'i')
    counts = None
    for i in range(maxiter):
        outs = array([argmindist2(x,means) for x in data],'i')
        mins = array(outs[:,0],'i')
        dists = outs[:,1]
        threshold = outlier * stats.scoreatpercentile(dists,per=50)
        changed = sum(mins!=oldmins)
        print changed
        if verbose: sys.stderr.write("[kmeans iter %d: %d]\n"%(i,changed))
        if changed<minchange: break
        avgdists = zeros(k)
        for i in range(k):
            where = ((mins==i) & (dists<threshold))
            avgdists[i] = mean(dists[where])
        if counts is not None: reuse = argsort(-counts)
        else: reuse = argsort(avgdists)
        j = 0
        counts = zeros(k)
        for i in range(k):
            where = ((mins==i) & (dists<threshold))
            counts[i] = sum(where)
            if counts[i]<minvecs: 
                print "%d=%d"%(i,reuse[j]),
                means[i] = means[reuse[j]]*(1.0+1e-5*randn(len(means[j])))
                j += 1
            else:
                means[i] = average(data[where],axis=0)
        print
        oldmins = mins
    return means,counts

