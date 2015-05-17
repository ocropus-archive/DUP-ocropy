from __future__ import absolute_import, division, print_function

from ocrolib.native import *

lstm_utils = r"""
#include <math.h>

void sumouter(int r,int n,int m,double out[n][m],double u[r][n],double v[r][m]) {
    for(int i=0;i<n;i++) {
        for(int j=0;j<m;j++) {
            double total = 0.0;
            for(int k=0;k<r;k++) total += u[k][i]*v[k][j];
            out[i][j] = total;
        }
    }
}

void sumprod(int r,int n,double out[n],double u[r][n],double v[r][n]) {
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int k=0;k<r;k++) total += u[k][i]*v[k][i];
        out[i] = total;
    }
}
"""

lstm_native = compile_and_load(lstm_utils)
lstm_native.sumouter.argtypes = [I,I,I,A2D,A2D,A2D]
lstm_native.sumprod.argtypes = [I,I,A1D,A2D,A2D]

def sumouter(u,v,out=None):
    assert out.shape==u.shape[1:]+v.shape[1:] and u.shape[:1]==v.shape[:1]
    lstm_native.sumouter(u.shape[0],out.shape[0],out.shape[1],out,u,v)
    return out
def sumprod(u,v,out=None):
    assert out.shape==u.shape[1:] and out.shape==v.shape[1:] and u.shape[:1]==v.shape[:1]
    lstm_native.sumprod(len(u),len(out),out,u,v)
    return out

def test():
    from pylab import randn
    sumouter(randn(11,3),randn(11,4),out=randn(3,4))
    sumprod(randn(11,7),randn(11,7),out=randn(7))
