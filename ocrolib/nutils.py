from ocrolib.native import *

lstm_utils = r"""
#include <math.h>

void sigmoid(int n,double v[n],double out[n]) {
    for(int i=0;i<n;i++) {
        double x = v[i];
        x = x<-100?-100:x>100?100:x;
        out[i] = 1.0/(1.0+exp(-x));
    }
}

void dotplus(int n,int m,double v[n],double a[n][m],double u[m]) {
    for(int i=0;i<n;i++) {
        double total;
        for(int j=0;j<m;j++) total += a[i][j]*u[j];
        v[i] += total;
    }
}

void prodplus(int n,double u[n],double v[n],double out[n]) {
    for(int i=0;i<n;i++) {
        out[i] += u[i]*v[i];
    }
}

void sumouter(int r,int n,int m,double a[n][m],double u[r][n],double v[r][m]) {
    for(int i=0;i<n;i++) {
        for(int j=0;j<m;j++) {
            double total = 0.0;
            for(int k=0;k<r;k++) total += u[k][i]*v[k][j];
            a[i][j] = total;
        }
    }
}

void sumprod(int r,int n,double u[r][n],double v[r][n],double a[n]) {
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int k=0;k<r;k++) total += u[k][i]*v[k][i];
        a[i] = total;
    }
}
"""

lstm_native = compile_and_load(lstm_utils)
lstm_native.sigmoid.argtypes = [I,A1D,A1D]
lstm_native.dotplus.argtypes = [I,I,A1D,A2D,A1D]
lstm_native.prodplus.argtypes = [I,A1D,A1D,A1D]
lstm_native.sumouter.argtypes = [I,I,I,A2D,A2D,A2D]
lstm_native.sumprod.argtypes = [I,I,A1D,A2D,A2D]

def sigmoid(u,out=None):
    assert u.shape==out.shape and len(u.shape)==1
    lstm_native.sigmoid(len(u),u,out)
    return out
def prodplus(u,v,out=None):
    assert u.shape==v.shape and u.shape==out.shape and len(u.shape)==1
    lstm_native.prodplus(len(out),out,u,v)
    return out
def sumouter(u,v,out=None):
    assert out.shape==u.shape[1:]+v.shape[1:] and u.shape[:1]==v.shape[:1]
    lstm_native.sumouter(u.shape[0],out.shape[0],out.shape[1],out,u,v)
    return out
def sumprod(u,v,out=None):
    assert out.shape==u.shape[1:] and out.shape==v.shape[1:] and u.shape[:1]==v.shape[:1]
    lstm_native.sumprod(len(u),len(out),out,u,v)
    return out

def test():
    from pylab import arange,zeros,randn
    sigmoid(randn(1000),out=randn(1000))
    dotplus(randn(3,4),randn(4),out=randn(3))
    prodplus(randn(3),randn(3),out=randn(3))
    sumouter(randn(11,3),randn(11,4),out=randn(3,4))
    sumprod(randn(11,7),randn(11,7),out=randn(7))
