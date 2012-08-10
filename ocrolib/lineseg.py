from pylab import *
from scipy.ndimage import filters,morphology,measurements
import common,morph
from toplevel import *

@checks(AFLOAT2,alpha=RANGE(0.0,20.0),r=RANGE(0,20))
def dpcuts(image,alpha=0.5,r=2):
    costs = 9999*ones(image.shape)
    costs[0,:] = 0
    sources = zeros(image.shape,'i')
    for i in range(1,len(costs)):
        for k in range(-r,r+1):
            ncosts = roll(costs[i-1,:],k)+image[i,:]+alpha*abs(k)
            sources[i,:] = where(ncosts<costs[i,:],-k,sources[i,:])
            costs[i,:] = where(ncosts<costs[i,:],ncosts,costs[i,:])
    return costs,sources

def between(u,v):
    u,v = min(u,v),max(u,v)
    for i in range(u,v+1):
        yield i

def dptrack(l,s):
    result = zeros(s.shape)
    for i in l:
        x0 = i
        x = i
        y = len(s)-1
        while y>-1:
            x = clip(x,0,result.shape[1]-1)
            for j in between(x0,x):
                result[y,j] = 1
            y -= 1
            x0 = x
            x += s[y,x]
    return result

@checks(AFLOAT2,imweight=RANGE(-20,20),bweight=RANGE(-20,20),diagweight=RANGE(-20,20))
def dplineseg1(image,imweight=4,bweight=-1,diagweight=1):
    cimage = imweight*image - bweight*maximum(0,roll(image,-1,1)-image)
    c,s = dpcuts(cimage,alpha=diagweight)
    costs = c[-1]
    costs = filters.gaussian_filter(costs,1)
    mins = find(filters.minimum_filter(costs,8)==costs)
    tracks = dptrack(mins,s)
    # combo = 3*tracks+cimage
    return tracks

@checks(AFLOAT2)
def centroid(image):
    ys,xs = mgrid[:image.shape[0],:image.shape[1]]
    yc = sum(image*ys)/sum(image)
    xc = sum(image*xs)/sum(image)
    return yc,xc

@checks(AFLOAT2,imweight=RANGE(-20,20),bweight=RANGE(-20,20),diagweight=RANGE(-20,20),r=RANGE(0,4),debug=BOOL)
def dplineseg2(image,imweight=4,bweight=-1,diagweight=1,r=2,debug=0):
    yc,xc = centroid(image)
    half = int(yc)
    cimage = imweight*image-bweight*maximum(0,roll(image,-1,1)-image)
    tc,ts = dpcuts(cimage[:half],alpha=diagweight,r=r)
    bc,bs = dpcuts(cimage[half:][::-1],alpha=diagweight,r=r)
    costs = bc[-1]+tc[-1]
    if debug:
        clf()
        subplot(311); imshow(tc)
        subplot(312); imshow(bc)
    costs = tc[-1]+bc[-1]
    costs = -costs
    costs -= amin(costs)
    costs = filters.gaussian_filter(costs,1)
    costs += 0.01*filters.gaussian_filter(costs,3.0)
    mins = (filters.maximum_filter(costs,8)==costs)*(costs>0.3*amax(costs))
    l = find(mins)
    tt = dptrack(l,ts)
    bt = dptrack(l,bs)
    tracks = r_[tt,bt[::-1]]
    if debug:
        subplot(313)
        imshow(tracks+0.5*image,interpolation='nearest')
    return tracks

@checks(LIGHTLINE)
def ccslineseg(image):
    image = 1.0*(image>0.3*amax(image))
    sigma = 10.0
    smooth = filters.gaussian_filter(image,(sigma,3.0*sigma))
    center = (smooth==amax(smooth,axis=0)[newaxis,:])
    center = filters.maximum_filter(center,(3,3))
    center = morph.keep_marked(image>0.5,center)
    center = filters.maximum_filter(center,(2,2))
    center,_ = morph.label(center)
    center = morph.spread_labels(center)
    center *= image
    return center

class SimpleParams:
    def info(self,depth=0):
        """Print information about this object."""
        pass
    def pexists(self,name):
        """Check whether parameter NAME exists."""
        return name in dir(self)
    def pset(self,name,value):
        """Set parameter NAME to VALUE."""
        assert name in dir(self)
        self.__dict__[name] = value
    def pget(self,name):
        """Get the value of string parameter NAME."""
        return self.__dict__.get(name)
    def pgetf(self,name):
        """Get the value of floating point parameter NAME."""
        return float(self.__dict__.get(name))

import common as ocrolib

class CCSSegmentLine(SimpleParams):
    @checks(object,LIGHTLINE)
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        line = (line<0.5*(amax(line)+amin(line)))
        seg = ccslineseg(line)
        seg = morph.renumber_by_xcenter(seg)
        return seg
    
class DPSegmentLine(SimpleParams):
    @checks(object,ledge=RANGE(-10,10),imweight=RANGE(-10,10),bweight=RANGE(-10,10),
            diagweight=RANGE(-10,10),r=RANGE(1,100),debug=BOOL)
    def __init__(self,ledge=-0.1,imweight=4,bweight=-1,diagweight=0.3,r=1,debug=0):
        self.r = r
        self.imweight = imweight
        self.bweight = bweight
        self.diagweight = diagweight
        self.debug = debug
        self.ledge = ledge
    @checks(object,LIGHTLINE)
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        assert mean(line)>0.5*amax(line)
        line = amax(line)-line
        line = line+self.ledge*maximum(0,roll(line,-1,1)-line)
        tracks = dplineseg2(line,imweight=self.imweight,bweight=self.bweight,
                            diagweight=self.diagweight,debug=self.debug,r=self.r)
        tracks = array(tracks<0.5*amax(tracks),'i')
        tracks,_ = morph.label(tracks)
        self.tracks = tracks
        rsegs = morph.spread_labels(tracks)
        rsegs = rsegs*(line>0.5*amax(line))
        return morph.renumber_by_xcenter(rsegs)
