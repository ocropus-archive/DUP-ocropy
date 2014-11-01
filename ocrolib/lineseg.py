from pylab import *
from scipy.ndimage import filters,morphology,measurements
import morph
from toplevel import *

@checks(AFLOAT2,alpha=RANGE(0.0,20.0),r=RANGE(0,20))
def dpcuts(image,alpha=0.5,r=2):
    """Compute dynamic programming cuts through an image.
    The image contains the costs themselves, `alpha` is the
    cost of taking a diagonal step, and `r` is the range
    of diagonal steps to be considered (determining the
    maximum slope of a cut."""
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
    """Iterate over the values between `u` and `v`, inclusive."""
    u,v = min(u,v),max(u,v)
    for i in range(u,v+1):
        yield i

def dptrack(l,s):
    """Given a list `l` of starting locations and an
    image `s` of steps produced by `dpcuts`, trace the cuts
    and output an image containing the cuts. The output
    image is guaranteed to be partitioned into separate
    regions by the cuts (so that it can be labeled)."""
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
    """A dynamic programming line segmenter.  This computes cuts going from bottom
    to top.  It is only used for testing and is not recommended for actual use because
    these kinds of cuts do not work very well."""
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
    """Compute the centroid of an image."""
    # FIXME just use the library function
    ys,xs = mgrid[:image.shape[0],:image.shape[1]]
    yc = sum(image*ys)/sum(image)
    xc = sum(image*xs)/sum(image)
    return yc,xc

@checks(AFLOAT2,imweight=RANGE(-20,20),bweight=RANGE(-20,0),diagweight=RANGE(-20,20),r=RANGE(0,4),debug=BOOL)
def dplineseg2(image,imweight=4,bweight=-1,diagweight=1,r=2,debug=0,width=-1,wfactor=1.0,
               threshold=0.5,sigma=1.0):
    """Perform a dynamic programming line segmentation, as described in Breuel (1994).
    This computes best cuts going out from the center in both directions, then finds
    the loally minimum costs.  Paths that move diagonally are penalized, and paths
    that move along the left edge of a line are rewarded.  Paths can only occur
    separated by a minimum distance of `width`.  If `width` is `-1`, the width is
    estimated as `wfactor` times the square root of the second moment in the 
    `y` direction of the text line."""
    if width<0:
        import lineest
        width = wfactor*lineest.vertical_stddev(image)[0]
    yc,xc = centroid(image)
    half = int(yc)
    deriv = maximum(0,filters.gaussian_filter(image,(2,2),order=(0,1)))
    deriv /= amax(deriv)
    cimage = where(image,imweight*image,bweight*deriv)
    if debug:
        figure("debug-dpseg-costs")
        clf()
        subplot(411); imshow(cimage)
    tc,ts = dpcuts(cimage[:half],alpha=diagweight,r=r)
    bc,bs = dpcuts(cimage[half:][::-1],alpha=diagweight,r=r)
    costs = bc[-1]+tc[-1]
    if debug:
        figure("debug-dpseg-costs")
        subplot(412); imshow(tc)
        subplot(413); imshow(bc)
    costs = tc[-1]+bc[-1]
    costs = filters.gaussian_filter(costs,sigma)
    costs += 0.01*filters.gaussian_filter(costs,3.0*sigma)
    # costs -= amin(costs)
    mins = (filters.minimum_filter(costs,width)==costs) # *(costs>0.3*amax(costs))
    mins *= costs<threshold*median(abs(costs))
    if debug:
        figure("debug-dpseg-mins")
        plot(costs)
        plot(tc[-1])
        plot(bc[-1])
        plot(mins)
    l = find(mins)
    tt = dptrack(l,ts)
    bt = dptrack(l,bs)
    tracks = r_[tt,bt[::-1]]
    if debug:
        figure("debug-dpseg-costs")
        subplot(414)
        imshow(tracks+0.5*image,interpolation='nearest')
        ginput(1,0.1)
    return tracks

@checks(DARKLINE)
def ccslineseg(image,debug=0):
    image = 1.0*(image>0.3*amax(image))
    sigma = 10.0
    smooth = filters.gaussian_filter(image,(sigma,1.0*sigma),mode='constant')
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
    """Perform a dynamic programming line segmentation, as described in Breuel (1994).
    This computes best cuts going out from the center in both directions, then finds
    the loally minimum costs.  Paths that move diagonally are penalized, and paths
    that move along the left edge of a line are rewarded."""
    @checks(object,imweight=RANGE(0,10),bweight=RANGE(-10,0),diagweight=RANGE(0,10),r=RANGE(1,5),debug=BOOL)
    def __init__(self,imweight=4,bweight=-1,diagweight=1,r=1,debug=0,threshold=0.5):
        self.r = r
        self.imweight = imweight
        self.bweight = bweight
        self.diagweight = diagweight
        self.debug = debug
        self.threshold = threshold
    @checks(object,LIGHTLINE)
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        assert mean(line)>0.5*amax(line)
        line = amax(line)-line
        # line = line+self.ledge*maximum(0,roll(line,-1,1)-line)
        tracks = dplineseg2(line,imweight=self.imweight,bweight=self.bweight,
                            diagweight=self.diagweight,debug=self.debug,r=self.r,
                            threshold=self.threshold)
        tracks = array(tracks<0.5*amax(tracks),'i')
        tracks,_ = morph.label(tracks)
        self.tracks = tracks
        stracks = morph.spread_labels(tracks)
        rsegs = stracks*(line>0.5*amax(line))
        if 0:
            figure("temp")
            subplot(311); morph.showlabels(tracks)
            subplot(312); morph.showlabels(stracks)
            subplot(313); morph.showlabels(rsegs)
            raw_input()
        return morph.renumber_by_xcenter(rsegs)



def seq2list(seq,result=None):
    """Given an OpenCV sequence object representing contours,
    returns a list of 2D point arrays."""
    if result is None: result = []
    while seq:
        l = list(seq)
        result.append(array(l,'i'))
        # sub = seq2list(seq.v_next(),result)
        seq = seq.h_next()
    result.sort(key=len,reverse=1)
    return result

def image2contours(image,inside=0):
    import cv
    """Given an image, return a list of (n,2) arrays corresponding to the
    contours of that image. Uses OpenCV's FindContours with no approximation
    and finding both inside and outside contours.  The result is returned
    in decreasing length."""
    image = cv.fromarray(array(image>0,'B'))
    storage = cv.CreateMemStorage()
    if inside: mode = cv.CV_RETR_CCOMP
    else: mode = cv.CV_RETR_EXTERNAL
    seq = cv.FindContours(image, storage, mode, cv.CV_CHAIN_APPROX_NONE)
    del storage
    return seq2list(seq)

from scipy.spatial import distance
from scipy import linalg

def image_draw_line(image,y0,x0,y1,x1):
    d = ((y0-y1)**2+(x0-x1)**2)**.5
    for l in linspace(0.0,1.0,int(2*d+1)):
        image[int(l*y0+(1-l)*y1),int(l*x0+(1-l)*x1)] = 1

def contourcuts(image,maxdist=15,minrange=10,mincdist=20,sigma=1.0,debug=0,r=8,s=0.5):
    if debug:
        figure(1); clf(); imshow(image)

    # start by computing the contours
    contours = image2contours(image!=0)

    # generate a mask for grayscale morphology
    mask = s*ones((r,r))
    mask[2:-2,2:-2] = 0

    cuts = []

    # now handle each (external) contour individually
    for k,cs in enumerate(contours):
        # compute a matrix of all the pairwise distances of pixels
        # around the contour, then smooth it a little
        ds = distance.cdist(cs,cs)
        ds = filters.gaussian_filter(ds,(sigma,sigma),mode='wrap')
        # compute a circulant matrix telling us the pathlength
        # between any two pixels on the contour
        n = len(cs)
        l = abs(arange(n)-n/2.0)
        l = l[0]-l
        cds = linalg.circulant(l)
        
        # find true local minima (exclude ridges) by using the
        # structuring element above
        ge = morphology.grey_erosion(ds,structure=mask,mode='wrap')
        locs = (ds<=ge)

        # restrict it to pairs of points that are closer than maxdist
        locs *= (ds<maxdist)

        # restrict it to paris of points that are separated by
        # at least mincdist on the contour
        locs *= (cds>=mincdist)

        # label the remaining minima and locate them
        locs,n = measurements.label(locs)
        cms = measurements.center_of_mass(locs,locs,range(1,n+1))

        # keep only on of each pair (in canonical ordering)
        cms = [(int(i+0.5),int(j+0.5)) for i,j in cms if i<j]
        for i,j in cms:
            x0,y0 = cs[i]
            x1,y1 = cs[j]
            # keep only the near vertical ones
            if abs(y1-y0)>abs(x1-x0):
                color = 'r'
                cuts.append((cs[i],cs[j]))
            else:
                color = 'b'
            if debug:
                print (x0,y0),(x1,y1)
                figure(1); plot([x0,x1],[y0,y1],color)

        if debug:
            figure(2); clf(); ion(); imshow(locs!=0)
            figure(3); clf(); imshow(minimum(ds,maxdist*1.5),interpolation='nearest')
            ginput(1,0.1)
            print "hit ENTER"; raw_input()
    # now construct a cut image
    cutimage = zeros(image.shape)
    for ((x0,y0),(x1,y1)) in cuts:
        image_draw_line(cutimage,y0,x0,y1,x1)
    cutimage = filters.maximum_filter(cutimage,(3,3))
    if debug:
        figure(4); clf(); imshow(maximum(0,image-0.5*cutimage))
    return cutimage
        


class ComboSegmentLine(SimpleParams):
    """Perform a dynamic programming line segmentation, as described in Breuel (1994).
    This computes best cuts going out from the center in both directions, then finds
    the loally minimum costs.  Paths that move diagonally are penalized, and paths
    that move along the left edge of a line are rewarded.

    In addition, this segmenter also computes contour-based cuts; these handle
    cases like two touching "oo" that are not handled well by the dynamic programming
    cuts.  To integrate the two segmenters, the contour-based cuts are applied first,
    and then the dynamic programming algorithm; this ensures that the two strategies
    give consistent segmentations."""
    @checks(object,imweight=RANGE(0,10),bweight=RANGE(-10,0),diagweight=RANGE(0,10),r=RANGE(1,5),debug=BOOL,maxdist=RANGE(0,1000),minrange=RANGE(0,1000),mincdist=RANGE(0,1000),sigma=RANGE(0.0,100.0),rr=RANGE(3,300),s=RANGE(0.0,500.0))
    def __init__(self,imweight=4,bweight=-1,diagweight=1,r=1,debug=0,
                 maxdist=15,minrange=10,mincdist=20,sigma=1.0,rr=8,s=0.5):
        self.r = r
        self.imweight = imweight
        self.bweight = bweight
        self.diagweight = diagweight
        self.debug = debug
        self.maxdist = maxdist
        self.minrange = minrange
        self.mincdist = mincdist
        self.sigma = sigma
        self.rr = rr
        self.s = s
    @checks(object,LIGHTLINE)
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        assert mean(line)>0.5*amax(line)
        line0 = amax(line)-line
        ccuts = contourcuts(line0,maxdist=self.maxdist,minrange=self.minrange,
                            mincdist=self.mincdist,sigma=self.sigma,r=self.rr,s=self.s)
        line = maximum(0,line0-ccuts)
        # line = line+self.ledge*maximum(0,roll(line,-1,1)-line)
        tracks = dplineseg2(line,imweight=self.imweight,bweight=self.bweight,
                            diagweight=self.diagweight,debug=self.debug,r=self.r)
        tracks = array(tracks<0.5*amax(tracks),'i')
        tracks,_ = morph.label(tracks)
        self.tracks = tracks
        stracks = morph.spread_labels(tracks)
        rsegs = stracks*(line0>0.5*amax(line0))
        if self.debug:
            figure("temp")
            subplot(311); morph.showlabels(tracks)
            subplot(312); morph.showlabels(stracks)
            subplot(313); morph.showlabels(rsegs)
            raw_input()
        return morph.renumber_by_xcenter(rsegs)


### A top-level driver for quick and simple testing.

if __name__=="__main__":
    import argparse
    parser= argparse.ArgumentParser("Testing line segmentation models.")
    subparsers = parser.add_subparsers(dest="subcommand")
    test = subparsers.add_parser("test")
    test.add_argument("--imweight",type=float,default=4,help="image weight (%(default)f")
    test.add_argument("--bweight",type=float,default=-1,help="left border weight (%(default)f)")
    test.add_argument("--diagweight",type=float,default=1,help="additional diagonal weight (%(default)f)")
    test.add_argument("--r",type=int,default=1,help="range for diagonal steps (%(default)d)")
    test.add_argument("--threshold",type=float,default=0.5)
    test.add_argument("files",nargs="+",default=[])
    # test2 = subparsers.add_parser("test2")
    args = parser.parse_args()
    if args.subcommand=="test":
        segmenter = DPSegmentLine(imweight=args.imweight,
                                  bweight=args.bweight,
                                  diagweight=args.diagweight,
                                  r=args.r,
                                  threshold=args.threshold,
                                  debug=1)
        ion(); gray()
        for fname in args.files:
            print fname
            image = ocrolib.read_image_gray(fname)
            segmentation = segmenter.charseg(image)
            figure("output")
            subplot(211); imshow(image)
            subplot(212); morph.showlabels(segmentation)
            raw_input()
        else:
            parser.print_help()
    sys.exit(0)

