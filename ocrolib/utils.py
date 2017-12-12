import numpy as np

def sumouter(u,v,out=None):
    if out is None:
        m = u.shape[1]
        n = v.shape[1]
        out = np.zeros((m,n))
    return np.einsum('ki,kj->ij',u,v,out=out)
    
def sumprod(u,v,out=None):
    if out is None:
        n = u.shape[1]
        out = np.zeros(n)
    return np.einsum('ki,ki->i',u,v,out=out)
