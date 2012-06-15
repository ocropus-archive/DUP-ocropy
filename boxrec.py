################################################################
### Segmenting line recognizer.  This is the main recognizer
### in OCRopus right now.
################################################################

import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from pylab import randn
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology,filters
from pylab import *
from ocrolib import sl

import cPickle as pickle
pickle_mode = 2

with open("/usr/local/share/ocropus/uw3unlv-240-8-40-nogeo.cmodel") as stream:
    cmodel = pickle.load(stream)



test = 1
debug = 0

def ain(image,l):
    """Array-in; returns a boolean array with the same shape as image for
    every pixel value in image that is contained in l."""
    return in1d(image.ravel(),l.ravel()).reshape(image.shape)

def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,return_distances=1,return_indices=1)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread

def multi_projection_mask(image,threshold=1,maxskew=0.5,steps=30):
    mask = ones(image.shape)
    for s in linspace(0.0,maxskew,steps):
        c = array(image.shape)/2
        m = array([[1,0],[-s,1]])
        image1 = interpolation.affine_transform(image,m,c-dot(m,c),order=0)
        mask1 = ones(image.shape)
        proj1 = sum(image1,axis=0)
        mask1[:,proj1<threshold] = 0
        mi = inv(m)
        mask1 = interpolation.affine_transform(mask1,mi,c-dot(mi,c),order=0)
        mask = minimum(mask,mask1)
    return mask

def estimate_lscale(binary):
    labels,_ = measurements.label(binary>0.5)
    scales = [sl.height(o) for o in measurements.find_objects(labels)]
    scales = [s for s in scales if s>3]
    return median(scales)

def keep_marked(image,markers):
    labels,_ = measurements.label(image)
    marked = unique(labels*(markers!=0))[1:]
    kept = in1d(labels.ravel(),marked)
    return kept.reshape(*labels.shape)

def keep_center(image,width=3):
    scale = estimate_lscale(image)
    blur = filters.gaussian_filter(1.0*image,(scale/2,scale/2),mode='constant')
    blur = filters.uniform_filter(blur,(1,3*scale),mode='constant')
    center = (blur>=amax(blur,axis=0)[newaxis,:])*(blur>0.01*amax(blur))
    center = filters.maximum_filter(center,(width,width))
    return keep_marked(image,center)

if 0:
    imshow(keep_center(image))

def character_labels(image,width=3):
    """Segment a line into potential characters, mostly by connected
    components.  This first finds all the components that intersect
    the center of the line.  Then, it assigns other components
    based on their overlap with the components of the center line."""
    # TODO use local orientation to help with idots on italics
    center = keep_center(image,width=width)
    if debug: imshow(center)
    clabels,_ = measurements.label(center)
    empty = (slice(9999,9999),slice(9999,9999))
    cobjects = [empty]+measurements.find_objects(clabels)
    llabels,_ = measurements.label(image)
    lobjects = [empty]+measurements.find_objects(llabels)
    selection = zeros(len(lobjects),'i')
    for j in range(1,len(cobjects)):
        c = cobjects[j]
        for i in range(1,len(lobjects)):
            l = lobjects[i]
            if sl.center_in(l,c):
                selection[i] = j
            elif l[0].stop<c[0].start and sl.xoverlaps(l,c):
                selection[i] = j
    rlabels = selection[llabels]
    return rlabels

def renumber(x):
    where(x==0,0.0,0.1+sin(x)**2)

if 0:
    rlabels = character_labels(image)
    imshow(rlabels,cmap=cm.flag)

def skew_image(image,s):
    """Skew a line image horizontally by the given amount.
    Italics are generally around s=0.3"""
    c = array(image.shape)/2
    m = array([[1,0],[-s,1]])
    return interpolation.affine_transform(image,m,c-dot(m,c),order=0)

def line_skew(image):
    """Find the skew of a line image by maximizing the variance
    of the skewed projection."""
    vs = []
    for s in linspace(0,0.5,30):
        sk = skew_image(image,s)
        pr = sum(sk,axis=0)
        v = var(pr)
        vs.append((v,s))
    # clf(); plot([y for x,y in vs],[x for x,y in vs])
    _,skew = max(vs)
    return skew

def segment_line(line,sigma=1.0,hrange=10):
    """Compute a line segmentation into components by finding
    minima of the projection.  The segmentation is performed for
    the skew that maximizes the variance of the projection.
    hrange should generally be 1.5*scale"""
    skew = line_skew(line)
    sk = skew_image(line,skew)
    pr = sum(sk,axis=0)
    pr = filters.gaussian_filter(pr,sigma)
    cut = (pr<=filters.minimum_filter(pr,hrange))
    cutimage = ones(line.shape)
    cutimage[:,cut] = 0
    cutimage = skew_image(cutimage,-skew)
    cutimage,_ = measurements.label(cutimage)
    cutimage = spread_labels(cutimage)
    segmentation = array(cutimage*line,'i')
    return segmentation

if 0:
    segmentation = segment_line(line)
    clf(); imshow(segmentation,cmap=cm.prism)

class Gridder:
    def __init__(self,rows=6,cols=12):
        self.rows = rows
        self.cols = cols
        self.count = 0
    def clear(self):
        self.count = 0
        clf()
    def show(self,image,label=None,cmap=cm.gray):
        if debug: 
            subplot(self.rows,self.cols,self.count%(self.rows*self.cols)+1)
            xticks([])
            yticks([])
            if label is not None: xlabel(label)
            imshow(image,cmap=cmap)
        self.count += 1
    def next(self):
        if not debug:
            subplot(self.rows,self.cols,self.count%(self.rows*self.cols)+1)
        self.count += 1

if test:
    gridder = Gridder()

class Boxrec:
    def __init__(self):
        self.line = None
        self.scale = None
        self.candidates = None
    def __str__(self):
        return "<Boxrec %s scale=%s #%s>"%(
            self.line.shape if self.line is not None else None,
            self.scale if self.scale is not None else None,
            len(self.candidates) if self.candidates is not None else None
            )
    def __repr__(self):
        return str(self)

class Candidate:
    __slots__ = "index label out bounds mask rsegs".split()
    def __init__(self,**kw):
        self.bounds = None
        self.out = None
        self.rsegs = []
        self.__dict__.update(kw)
    def __str__(self):
        return "<C '%s' %s (%s %s) %s>"%(
            self.out[0][0] if self.out else None,
            "%.2f"%self.out[0][1] if self.out else None,
            sl.width(self.bounds) if self.bounds else None,
            sl.height(self.bounds) if self.bounds else None,
            self.rsegs if self.rsegs is not None else []
            )
    def __repr__(self):
        return str(self)
    
def load(S,fname):
    image = imread(fname)
    if image.ndim==3: image = mean(image,2)
    image -= amin(image)
    image /= amax(image)
    image = (image<0.5)
    ion()
    gray()
    # image = image[:,1000:1500]
    line = interpolation.zoom(1.0*image,2.0,order=3)
    line = 1.0*(line>0.5)
    clf()
    if debug: imshow(line,interpolation='nearest')
    S.line = line
    S.scale = estimate_lscale(line)
    print "scale",S.scale
    return S

if test:
    S = Boxrec()
    load(S,"t/0001/010006.png")
    #load(S,"t/0001/010010.png")
    print "loaded",S

def raw_candidates(S):
    # compute the raw candidates

    S.rlabels = character_labels(S.line,width=int(1+S.scale/4))
    S.robjects = enumerate(measurements.find_objects(S.rlabels))
    S.robjects = [(i+1,o) for i,o in S.robjects]
    S.robjects = sorted(S.robjects,key=lambda x:sl.xcenter(x[1]))

    gridder.clear()

    # these are the raw segmentation S.candidates
    S.raw = []

    # these are all S.candidates that we have found
    S.candidates = []
    # construct a second segmentation map with the extra splits
    # due to split characters
    S.clabels = S.rlabels.copy()

    for q,(i,o) in enumerate(S.robjects):
        sub = S.line[o]*(S.rlabels[o]==i)
        out = cmodel.coutputs(sub)
        gridder.show(sub,label="%d %s"%(q,out[0][0]))
        r = Candidate(index=q,rsegs=[i],bounds=o,mask=S.rlabels[o]==i,out=out)
        S.candidates.append(r)
        S.raw.append(r)
    ginput(1,0.001)

if test:
    raw_candidates(S)
    print "raw",S

def join_candidates(S):
    gridder.clear()

    # compute pairs of rejected characters and see whether
    # they fit together

    clf()
    for i in range(len(S.raw)-1):
        if S.raw[i].out[0][0]!="~": continue
        if S.raw[i+1].out[0][0]!="~": continue
        o = sl.union(S.raw[i].bounds,S.raw[i+1].bounds)
        mask = ain(S.rlabels[o],unique(S.raw[i].rsegs+S.raw[i+1].rsegs))
        sub = S.line[o]*mask
        out = cmodel.coutputs(sub)
        gridder.show(sub,label="%s"%out[0][0])
        if out[0][0]!="~":
            r = Candidate(bounds=o,mask=mask,out=out,rsegs=[i,i+1])
            S.candidates.append(r)

if test:
    join_candidates(S)
    print "join",S

def split_candidates(S,split=0.8):
    """Find otherwise rejected components that may need to get split."""
    gridder.clear()
    for i in range(len(S.raw)):
        if S.raw[i].out[0][0]!="~": continue
        o = S.raw[i].bounds
        # don't bother if it's too small
        if sl.width(o)<split*S.scale: continue
        mask = ain(S.rlabels[o],unique(S.raw[i].rsegs))
        sub = S.line[o]*mask
        offset = (o[0].start,o[1].start)
        lsub = segment_line(sub,hrange=split*S.scale)
        S.clabels[o] += 1000*lsub
        osub = measurements.find_objects(lsub)
        n = amax(lsub)+1
        gridder.show(lsub,cmap=cm.jet,label="<%d>"%(n-1))
        for i in range(1,n):
            for j in range(i,n):
                if i==1 and j==n-1: continue
                if j-i>1: continue
                o = osub[i-1]
                if o is None: continue
                for k in range(i+1,j+1): o = sl.union(o,osub[k-1])
                frag = sub[o]
                mfrag = lsub[o]
                assert frag.ndim==2,"%s %s %s"%(frag.shape,o,sub.shape)
                out = cmodel.coutputs(frag)
                out = sorted(out,key=lambda x:-x[1])
                mask = ain(mfrag,arange(i,j+1))
                gridder.show(frag*mask,label="%s"%out[0][0])
                r = Candidate(bounds=sl.shift(o,offset),mask=mask,out=out)
                S.candidates.append(r)

if test:
    split_candidates(S)
    print "split",S

    
if 0:
    clf()
    imshow(sin(S.clabels)**2,cmap=cm.flag)

def renumber_labels(l):
    """Renumber the labels in an image so that they
    run from 1 to len(unique(l))."""
    u = unique(l.ravel())
    m = zeros(amax(u)+1,'i')
    m[sorted(u)] = arange(len(u),dtype='i')
    return m[l]

def sort_labels_x(l):
    """Renumbers the labels in l such that the corresponding
    bounding boxes are ordered by their x-center."""
    l = renumber_labels(l)
    objects = measurements.find_objects(l)
    centers = [sl.xcenter(o) for o in objects]
    print centers
    order = argsort(centers)
    indexes = arange(len(objects))
    result = zeros(len(objects))
    result[order] = indexes
    remap = r_[zeros(1,'i'),1+result]
    output = remap[l]
    return output

if 0:
    subplot(121); imshow(S.clabels,cmap=cm.jet)
    subplot(122); imshow(sort_labels_x(S.clabels),cmap=cm.jet)

def print_candidate_segs(S):
    for c in S.candidates:
        print unique(where(c.mask,temp[c.bounds],0))[1:]

def add_segs(S):
    S.clabels = sort_labels_x(S.clabels)
    for c in S.candidates:
        c.rsegs = array(unique(where(c.mask,S.clabels[c.bounds],0))[1:],'i')
    S.candidates.sort(key=lambda c:tuple(c.rsegs))
        
if test:
    add_segs(S)
    for c in S.candidates:
        print c

def overlap(cu,cv):
    """Given two character candidates cu and cv, computes
    the number of overlapping pixels.  This uses c.bounds and c.mask"""
    u,v = cu.bounds,cv.bounds
    if not sl.xoverlaps(u,v): return 0
    if not sl.yoverlaps(u,v): return 0
    delta = (-min(u[0].start,v[0].start),-min(u[1].start,v[1].start))
    u,v = sl.shift(u,delta),sl.shift(v,delta)
    full = sl.union(u,v)
    test = zeros((sl.height(full),sl.width(full)))
    test[u] += cu.mask
    test[v] += cv.mask
    n = sum(test>1)
    return n

def show_candidates(S):
    candidates = list(S.candidates)
    candidates.sort(key=lambda c:sl.xcenter(c.bounds))
    gridder.clear()
    for c in candidates:
        gridder.show(c.mask,label="%s"%c.out[0][0])

if test:
    show_candidates(S)



def is_neighbor(u,v,rest,minsize=10):
    uc = sl.xcenter(u.bounds)
    vc = sl.xcenter(v.bounds)
    if uc>=vc: return 0
    if overlap(u,v)>0.3*minimum(sum(u.mask),sum(v.mask)):
        return 0
    for w in rest:
        if sum(w.mask)<minsize: continue
        wc = sl.xcenter(w.bounds)
        if wc<=uc or wc>=vc: continue
        if overlap(u,w)>0: continue
        if overlap(w,v)>0: continue
        return 0
    return 1

def adjust_rejects(outputs,factor=0.01):
    def adj(s):
        c,p = s
        if c=="~": p *= factor
        return (c,p)
    outputs = [adj(s) for s in outputs]
    outputs.sort(key=lambda x:-x[1])
    return outputs

def show_neighbors(S):
    for i,u in enumerate(S.candidates):
        for j,v in enumerate(S.candidates):
            if not is_neighbor(u,v,S.candidates): continue
            print i,j,u,v

def show_graph(S):
    with open("fst.dot","w") as stream:
        stream.write("digraph q {\n")
        stream.write("rankdir=LR\n")
        # stream.write("node [shape=plaintext]\n")
        for i,u in enumerate(S.candidates):
            out = adjust_rejects(u.out)
            stream.write('a%d [label="%s"];\n'%(i,out[0][0]))
        minsize = (S.scale/4)**2
        print minsize
        for i,u in enumerate(S.candidates):
            for j,v in enumerate(S.candidates):
                if not is_neighbor(u,v,S.candidates,minsize=minsize): continue
                print i,j,u,v
                stream.write('a%d -> a%d\n'%(i,j))
        stream.write("}\n")
    os.system("dot -Tpng fst.dot -o fst.png")
    os.system("eog fst.png&")
