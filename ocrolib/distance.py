################################################################
### Native code neural network with backpropagation training.
################################################################

from __future__ import with_statement

__all__ = "MLP".split()

from numpy import *
from pylab import *
from scipy import *
from native import *
import multiprocessing

def c_order(a):
    """Check whether the elements of the array are in C order."""
    return tuple(a.strides)==tuple(sorted(a.strides,reverse=1))

cdist_native_c = r'''
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

int maxthreads = 1;

void cdist(int d,int na,int nb,float a[na][d],float b[nb][d],float result[na][nb]) {
    int n = na*nb;
#pragma omp parallel for num_threads (maxthreads)
    for(int job=0;job<n;job++) {
        int i = job/nb;
        int j = job%nb;
        double total = 0.0;
        for(int k=0;k<d;k++) {
            float delta = a[i][k]-b[j][k];
            total += delta*delta;
        }
        result[i][j] = sqrt(total);
    }
}
'''

cdist_native = None

def cdist_native_load():
    global cdist_native
    if cdist_native is not None: return
    cdist_native = compile_and_load(cdist_native_c)
    cdist_native.cdist.argtypes = [I,I,I,A2F,A2F,A2F]
    global maxthreads
    maxthreads = c_int.in_dll(cdist_native,"maxthreads")
    maxthreads.value = multiprocessing.cpu_count()

def cdist(a,b,out=None,threads=-1):
    cdist_native_load()
    if type(a)==list or a.dtype!=dtype('float32') or not c_order(a):
        a = array(a,dtype='float32',order="C")
    if type(b)==list or b.dtype!=dtype('float32') or not c_order(b):
        b = array(b,dtype='float32',order="C")
    assert a.ndim==2
    assert b.ndim==2
    assert a.shape[1]==b.shape[1]
    na = len(a)
    nb = len(b)
    d = a.shape[1]
    if out is None:
        out = zeros((len(a),len(b)),'float32')
    if threads<0: threads = multiprocessing.cpu_count()
    maxthreads.value = threads
    cdist_native.cdist(d,na,nb,a,b,out)
    return out

class ProtoDists:
    def __init__(self):
        pass
    def setProtos(self,b):
        assert b.ndim==2
        assert c_order(b)
        if b.dtype!=dtype('float32'): b = array(b,'float32')
        self.b = b
    def cdist(self,a):
        if a.dtype!=dtype('float32'): a = array(a,'float32')
        return cdist(a,self.b,threads=-1)

if __name__=="__main__":
    cdist_native_load()
    a = array(randn(3,5),'f')
    b = array(randn(7,5),'f')
    from scipy.spatial.distance import cdist as oldcdist
    out = cdist(a,b)
    print out
    out2 = oldcdist(a,b)
    print out2
