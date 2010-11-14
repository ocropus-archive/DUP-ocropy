
native_ocrofast = """
#include <omp.h>
#include <math.h>

int omp_nprocs() {
    return omp_get_num_procs();
}

int omp_nthreads() {
    return omp_get_max_threads();
}

void alldists(int n,int m,float out[n],float a[n][m],float v[m]) {
#pragma omp parallel for
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
        }
        out[i] = sqrt(total);
    }
}

void alldists_d(int n,int m,double out[n],double a[n][m],double v[m]) {
#pragma omp parallel for
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            // if(total>md) break;
        }
        out[i] = sqrt(total);
    }
}

void alldists_sig(int n,int m,float out[n],float a[n][m],float s[n][m],float v[m]) {
#pragma omp parallel for
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = (a[i][j]-v[j])/s[i][j];
            total += d*d;
        }
        out[i] = sqrt(total);
    }
}

void alldists_sig_d(int n,int m,double out[n],double a[n][m],double s[n][m],double v[m]) {
#pragma omp parallel for
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = (a[i][j]-v[j])/s[i][j];
            total += d*d;
            // if(total>md) break;
        }
        out[i] = sqrt(total);
    }
}

void alldists_sig1(int n,int m,float out[n],float a[n][m],float s[m],float v[m]) {
#pragma omp parallel for
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = (a[i][j]-v[j])/s[j];
            total += d*d;
        }
        out[i] = sqrt(total);
    }
}

void alldists_sig1_d(int n,int m,double out[n],double a[n][m],double s[m],double v[m]) {
#pragma omp parallel for
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = (a[i][j]-v[j])/s[j];
            total += d*d;
            // if(total>md) break;
        }
        out[i] = sqrt(total);
    }
}

void subsetdists(int r,int n,int m,float out[r],int subset[r],float a[n][m],float v[m]) {
#pragma omp parallel for
    for(int index=0;index<r;index++) {
        int i = subset[index];
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
        }
        out[index] = sqrt(total);
    }
}

void subsetdists_d(int r,int n,int m,double out[r],int subset[r],double a[n][m],double v[m]) {
#pragma omp parallel for
    for(int index=0;index<r;index++) {
        int i = subset[index];
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
        }
        out[index] = sqrt(total);
    }
}

int argmindist(int n,int m,float a[n][m],float v[m]) {
    int mi = -1;
    double md = 1e300;
#pragma omp parallel for shared(mi,md)
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            // if(total>md) break;
        }
#pragma omp critical
        if(total<md) {
            md = total;
            mi = i;
        }
    }
    return mi;
}

int argmindist_d(int n,int m,double a[n][m],double v[m]) {
    int mi = -1;
    double md = 1e300;
#pragma omp parallel for shared(mi,md)
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            // if(total>md) break;
        }
#pragma omp critical
        if(total<md) {
            md = total;
            mi = i;
        }
    }
    return mi;
}

int argmindist_sig(int n,int m,float a[n][m],float s[n][m],float v[m]) {
    int mi = -1;
    double md = 1e300;
#pragma omp parallel for shared(mi,md)
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            // if(total>md) break;
        }
#pragma omp critical
        if(total<md) {
            md = total;
            mi = i;
        }
    }
    return mi;
}

int argmindist_sig_d(int n,int m,double a[n][m],float s[n][m],double v[m]) {
    int mi = -1;
    double md = 1e300;
#pragma omp parallel for shared(mi,md)
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            // if(total>md) break;
        }
#pragma omp critical
        if(total<md) {
            md = total;
            mi = i;
        }
    }
    return mi;
}

int argmindist_nz(int n,int m,float a[n][m],float v[m],float low) {
    int mi = -1;
    double md = 1e300;
#pragma omp parallel for shared(mi,md)
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            // if(total>md) break;
        }
#pragma omp critical
        if(total>=low && total<md) {
            md = total;
            mi = i;
        }
    }
    return mi;
}

int epsfirst(int n,int m,float a[n][m],float v[m],float eps) {
    int mi = 9999999;
    float eps2 = eps*eps;
#pragma omp parallel for shared(mi)
    for(int i=0;i<n;i++) {
        if(i>mi) continue;
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
            if(total>=eps2) break;
        }
#pragma omp critical
        if(total<eps2) mi = i;
    }
    if(mi>n) return -1;
    return mi;
}

int dlfirst(int n,int m,float a[n][m],float eps[n],float v[m]) {
    int mi = 9999999;
#pragma omp parallel for shared(mi)
    for(int i=0;i<n;i++) {
        if(i>mi) continue;
        double total = 0.0;
        for(int j=0;j<m;j++) {
            double d = a[i][j]-v[j];
            total += d*d;
        }
#pragma omp critical
        if(total<eps[i]*eps[i]) mi = i;
    }
    return mi;
}

int argmindist_shift(int n,int r,float a[n][r][r],int s,float v[s][s]) {
    int mi = -1;
    double md = 1e300;
#pragma omp parallel for shared(mi,md)
    for(int i=0;i<n;i++) {
        for(int dx=0;dx<s-r;dx++) {
            for(int dy=0;dy<s-r;dy++) {
                double total = 0.0;
                for(int x=0;x<r;x++) for(int y=0;y<r;y++) {
                     double d = a[i][x][y]-v[x+dx][y+dy];
                     total += d*d;
                     // if(total>md) break;
                }
#pragma omp critical
                if(total<md) {
                    md = total;
                    mi = i;
                }
            }
        }
    }
    return mi;
}
"""

import numpy
from  numpy.ctypeslib import ndpointer
from ctypes import c_int,c_float,c_double
import ctypes
import timeit
from pylab import prod
from ocropy import native

# TODO:
# argmindist_nbest(a,v,n)
# subsetdists(a,v,subset)

ocrofast = native.compile_and_load(native_ocrofast)

ocrofast.omp_nprocs.restype = c_int
ocrofast.omp_nthreads.restype = c_int

ocrofast.alldists.restype = c_int
ocrofast.alldists.argtypes = [c_int,c_int,
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED')]

ocrofast.alldists_d.restype = c_int
ocrofast.alldists_d.argtypes = [c_int,c_int,
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED')]

def alldists(a,v,out=None):
    """Find the index of the row whose distance from v is smallest."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    if a.dtype==numpy.dtype('float32'):
        if out is None: out = numpy.zeros(n,'f')
        assert v.dtype==numpy.dtype('float32')
        assert out.dtype==numpy.dtype('float32')
        ocrofast.alldists(n,m,out,a,v)
        return out
    elif a.dtype==numpy.dtype('float64'):
        if out is None: out = numpy.zeros(n,'d')
        assert v.dtype==numpy.dtype('float64')
        assert out.dtype==numpy.dtype('float64')
        ocrofast.alldists_d(n,m,out,a,v)
        return out
    else:
        raise Exception("unknown data type")

ocrofast.alldists_sig.restype = c_int
ocrofast.alldists_sig.argtypes = [c_int,c_int,
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED')]

ocrofast.alldists_sig_d.restype = c_int
ocrofast.alldists_sig_d.argtypes = [c_int,c_int,
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED')]


def alldists_sig(a,s,v,out=None):
    """Find all the Mahalanobis distances of v from the rows of a, using s as a
    list of diagonal covariance matrices."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    assert s.shape==(n,m)
    if a.dtype==numpy.dtype('float32'):
        if out is None: out = numpy.zeros(n,'f')
        assert v.dtype==numpy.dtype('float32')
        assert out.dtype==numpy.dtype('float32')
        ocrofast.alldists_sig(n,m,out,a,s,v)
        return out
    elif a.dtype==numpy.dtype('float64'):
        if out is None: out = numpy.zeros(n,'d')
        assert v.dtype==numpy.dtype('float64')
        assert out.dtype==numpy.dtype('float64')
        ocrofast.alldists_sig_d(n,m,out,a,s,v)
        return out
    else:
        raise Exception("unknown data type")

ocrofast.alldists_sig1.restype = c_int
ocrofast.alldists_sig1.argtypes = [c_int,c_int,
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED')]

ocrofast.alldists_sig1_d.restype = c_int
ocrofast.alldists_sig1_d.argtypes = [c_int,c_int,
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED')]


def alldists_sig1(a,s,v,out=None):
    """Find all the Mahalanobis distances of v from the rows of a, using s as a
    single diagonal covariance matrix for all comparisons."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    assert s.shape==(m,)
    if a.dtype==numpy.dtype('float32'):
        if out is None: out = numpy.zeros(n,'f')
        assert v.dtype==numpy.dtype('float32')
        assert out.dtype==numpy.dtype('float32')
        ocrofast.alldists_sig1(n,m,out,a,s,v)
        return out
    elif a.dtype==numpy.dtype('float64'):
        if out is None: out = numpy.zeros(n,'d')
        assert v.dtype==numpy.dtype('float64')
        assert out.dtype==numpy.dtype('float64')
        ocrofast.alldists_sig1_d(n,m,out,a,s,v)
        return out
    else:
        raise Exception("unknown data type")

ocrofast.subsetdists.restype = c_int
ocrofast.subsetdists.argtypes = [c_int,c_int,c_int,
                                 ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='i',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                 ]

ocrofast.subsetdists_d.restype = c_int
ocrofast.subsetdists_d.argtypes = [c_int,c_int,c_int,
                                   ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                   ndpointer(dtype='i',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                   ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                   ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED')
                                   ]

def subsetdists(a,v,subset,out=None):
    """Find the index of the row whose distance from v is smallest."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    r = len(subset)
    if a.dtype==numpy.dtype('float32'):
        if out is None: out = numpy.zeros(n,'f')
        assert v.dtype==numpy.dtype('float32')
        assert out.dtype==numpy.dtype('float32')
        ocrofast.subsetdists(r,n,m,out,subset,a,v)
        return out
    elif a.dtype==numpy.dtype('float64'):
        if out is None: out = numpy.zeros(n,'d')
        assert v.dtype==numpy.dtype('float64')
        assert out.dtype==numpy.dtype('float64')
        ocrofast.subsetdists_d(r,n,m,out,subset,a,v)
        return out
    else:
        raise Exception("unknown data type")

ocrofast.argmindist.restype = c_int
ocrofast.argmindist.argtypes = [c_int,c_int,
                               ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                               ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED')]

ocrofast.argmindist_d.restype = c_int
ocrofast.argmindist_d.argtypes = [c_int,c_int,
                                 ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                 ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED')]

def argmindist(a,v):
    """Find the index of the row whose distance from v is smallest."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    if a.dtype==numpy.dtype('float32'):
        assert v.dtype==numpy.dtype('float32')
        return ocrofast.argmindist(n,m,a,v)
    elif a.dtype==numpy.dtype('float64'):
        assert v.dtype==numpy.dtype('float64')
        return ocrofast.argmindist_d(n,m,a,v)
    else:
        raise Exception("unknown data type")

ocrofast.argmindist_sig.restype = c_int
ocrofast.argmindist_sig.argtypes = [c_int,c_int,
                                    ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                    ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                    ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED')]

ocrofast.argmindist_sig_d.restype = c_int
ocrofast.argmindist_sig_d.argtypes = [c_int,c_int,
                                      ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                      ndpointer(dtype='d',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                      ndpointer(dtype='d',ndim=1,flags='CONTIGUOUS,ALIGNED')]

def argmindist_sig(a,s,v):
    """Find the index of the row whose distance from v is smallest."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    assert s.shape==(n,m)

    if a.dtype==numpy.dtype('float32'):
        assert v.dtype==numpy.dtype('float32')
        return ocrofast.argmindist_sig(n,m,a,s,v)
    elif a.dtype==numpy.dtype('float64'):
        assert v.dtype==numpy.dtype('float64')
        return ocrofast.argmindist_sig_d(n,m,a,s,v)
    else:
        raise Exception("unknown data type")

ocrofast.argmindist_nz.restype = c_int
ocrofast.argmindist_nz.argtypes = [c_int,c_int,
                                   ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                                   ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                                   c_float]

def argmindist_nz(a,v,low=1e-30):
    """Find the index of the row whose distance from v is smallest."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    if a.dtype==numpy.dtype('float32'):
        assert v.dtype==numpy.dtype('float32')
        return ocrofast.argmindist_nz(n,m,a,v,low)
    else:
        raise Exception("unknown data type")

ocrofast.epsfirst.restype = c_int
ocrofast.epsfirst.argtypes = [c_int,c_int,
                             ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                             ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                             c_float]

def epsfirst(a,v,eps):
    """Walk down the rows of a and find the first row whose
    Euclidean distance from v is smaller than eps."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    v = v.ravel()
    n,m = a.shape
    assert v.shape==(m,)
    if a.dtype==numpy.dtype('float32'):
        assert v.dtype==numpy.dtype('float32')
        return ocrofast.epsfirst(n,m,a,v,eps)
    else:
        raise Exception("unknown data type")

ocrofast.dlfirst.restype = c_int
ocrofast.dlfirst.argtypes = [c_int,c_int,
                             ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED'),
                             ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED'),
                             ndpointer(dtype='f',ndim=1,flags='CONTIGUOUS,ALIGNED')]

def dlfirst(a,eps,v):
    """Walk down the rows of a and find the first row whose
    Euclidean distance from v is smaller than the corresponding
    value in eps."""
    a = a.reshape(len(a),prod(a.shape[1:]))
    eps = eps.ravel()
    v = v.ravel()
    n,m = a.shape
    assert eps.shape==(n,)
    assert v.shape==(m,)
    if a.dtype==numpy.dtype('float32'):
        assert eps.dtype==numpy.dtype('float32')
        assert v.dtype==numpy.dtype('float32')
        return ocrofast.argmindist(n,m,a,eps,v)
    else:
        raise Exception("unknown data type")

ocrofast.argmindist_shift.restype = c_int
ocrofast.argmindist_shift.argtypes = [c_int,c_int,
                                      ndpointer(dtype='f',ndim=3,flags='CONTIGUOUS,ALIGNED'),
                                      c_int,
                                      ndpointer(dtype='f',ndim=2,flags='CONTIGUOUS,ALIGNED')]

def argmindist_shift(a,v):
    """Find the index of the row whose distance from v is smallest. Each
    row in a is an r x r image and v is an s x s image, with r<=v.  The
    Euclidean distance is computed for all shifts of the row image with
    the v image."""
    n,r,r2 = a.shape
    assert r==r2
    s,s2 = v.shape
    assert s==s2
    assert r<=s
    if a.dtype==numpy.dtype('float32'):
        assert v.dtype==numpy.dtype('float32')
        return ocrofast.argmindist(n,r,a,s,v)
    else:
        raise Exception("unknown data type")

def edist(u,v):
    import math
    from pylab import randn,newaxis,sum,argmin
    return math.sqrt(sum((u-v)**2))

def edist_sig(u,v,s):
    import math
    from pylab import randn,newaxis,sum,argmin
    return math.sqrt(sum(((u-v)/s)**2))

def argmindist_py(a,v):
    from pylab import randn,newaxis,sum,argmin
    return argmin(sum((a-v[newaxis,:])**2,axis=1))

def test():
    global a,v
    print "nprocs",ocrofast.omp_nprocs()
    print "nthreads",ocrofast.omp_nthreads()
    from pylab import randn,newaxis,sum,argmin
    a = randn(10000,100)
    v = randn(100)
    t = timeit.timeit(lambda:argmindist(a,v),number=100)
    result = argmindist(a,v)
    print "ocrofast",result,"time",t
    t = timeit.timeit(lambda:argmindist_py(a,v),number=100)
    result = argmindist_py(a,v)
    print "python",result,"time",t
