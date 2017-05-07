from scipy.ndimage import filters
from pylab import *
import re

def levenshtein(a,b):
    """Calculates the Levenshtein distance between a and b. 
    (Clever compact Pythonic implementation from hetland.org)"""
    n, m = len(a), len(b)
    if n > m: a,b = b,a; n,m = m,n       
    current = range(n+1)
    for i in range(1,m+1):
        previous,current = current,[i]+[0]*n
        for j in range(1,n+1):
            add,delete = previous[j]+1,current[j-1]+1
            change = previous[j-1]
            if a[j-1]!=b[i-1]: change = change+1
            current[j] = min(add, delete, change)
    return current[n]

def xlevenshtein(a,b,context=1):
    """Calculates the Levensthein distance between a and b
    and generates a list of differences by context."""
    n, m = len(a), len(b)
    assert m>0 # xlevenshtein should only be called with non-empty b string (ground truth)
    if a == b: return 0,[] # speed up for the easy case
    sources = empty((m+1,n+1),object)
    sources[:,:] = None
    dists = 99999*ones((m+1,n+1))
    dists[0,:] = arange(n+1)
    for i in range(1,m+1):
        previous = dists[i-1,:]
        current = dists[i,:]
        current[0] = i
        for j in range(1,n+1):
            if previous[j]+1<current[j]:
                sources[i,j] = (i-1,j)
                dists[i,j] = previous[j]+1
            if current[j-1]+1<current[j]:
                sources[i,j] = (i,j-1)
                dists[i,j] = current[j-1]+1
            delta = 1*(a[j-1]!=b[i-1])
            if previous[j-1]+delta<current[j]:
                sources[i,j] = (i-1,j-1)
                dists[i,j] = previous[j-1]+delta
    cost = current[n]

    # reconstruct the paths and produce two aligned strings
    l = sources[i,n]
    path = []
    while l is not None:
        path.append(l)
        i,j = l
        l = sources[i,j]
    al,bl = [],[]
    path = [(n+2,m+2)]+path
    for k in range(len(path)-1):
        i,j = path[k]
        i0,j0 = path[k+1]
        u = "_"
        v = "_"
        if j!=j0 and j0<n: u = a[j0]
        if i!=i0 and i0<m: v = b[i0]
        al.append(u)
        bl.append(v)
    al = "".join(al[::-1])
    bl = "".join(bl[::-1])

    # now compute a splittable string with the differences
    assert len(al)==len(bl)
    al = " "*context+al+" "*context
    bl = " "*context+bl+" "*context
    assert "~" not in al and "~" not in bl
    same = array([al[i]==bl[i] for i in range(len(al))],'i')
    same = filters.minimum_filter(same,1+2*context)
    als = "".join([al[i] if not same[i] else "~" for i in range(len(al))])
    bls = "".join([bl[i] if not same[i] else "~" for i in range(len(bl))])
    # print(als)
    # print(bls)
    ags = re.split(r'~+',als)
    bgs = re.split(r'~+',bls)
    confusions = [(a,b) for a,b in zip(ags,bgs) if a!="" or b!=""]
    return cost,confusions

