import copy as pycopy
from collections import Counter,namedtuple,defaultdict
from pylab import *
from scipy.ndimage import measurements,morphology
import common as ocrolib
from ocrolib import showgrid
import morph,lineseg,lineproc,sl

class Segment:
    def __init__(self,**kw):
        self.first = None
        self.last = None
        self.bbox = None
        self.img = None
        self.out = None
        self.sp = None
        assert set(kw.keys())<=set(self.__dict__.keys())
        self.__dict__.update(**kw)
    def replace(self,**kw):
        assert set(kw.keys())<=set(self.__dict__.keys())
        result = pycopy.copy(self)
        result.__dict__.update(**kw)
        return result
    def __str__(self):
        cls = self.out[0][0] if self.out is not None else ""
        return "<Segment %d:%d %s>"%(self.first,self.last,cls)
    def __repr__(self):
        return self.__str__()

def max_boxgap(boxes):
    gap = 0
    for i in range(1,len(boxes)):
        gap = maximum(gap,boxes[i][1].start-boxes[i-1][1].stop)
    return gap

def box_union(boxes):
    result = boxes[0]
    for i in range(1,len(boxes)):
        result = sl.union(result,boxes[i])
    return result

def extract_candidate_groups(segmentation,min_aspect=0.1,max_aspect=1.8,maxgap=2,maxrange=3):
    """Select a basic collection of candidate components.  Aspect ratios <1 are tall and skinny,
    >1 are wide.  `maxgap` is the maximum gap between bounding boxes.  `maxrange` gives the
    maximum number of compponents to be combined."""
    assert morph.ordered_by_xcenter(segmentation),"call morph.sort_by_xcenter first"
    boxes = [None]+morph.find_objects(segmentation)
    n = len(boxes)
    result = []
    for i in range(1,n):            
        for r in range(1,maxrange+1):
            if i+r>n: continue
            if r>1 and max_boxgap(boxes[i:i+r])>maxgap: 
                continue
            box = box_union(boxes[i:i+r])
            a = sl.aspect(box)
            if r>1 and 1.0/a>max_aspect: continue
            if 1.0/a<min_aspect: continue
            assert sum(segmentation[box])>0
            j = i+r-1
            seg = segmentation[box]*(segmentation[box]>=i)*(segmentation[box]<=j)
            result.append(Segment(first=i,last=j,bbox=box))
    return result

def extract_char(segmentation,g):
    seg = segmentation[:,g.bbox[1]]
    return 1*(seg>=g.first)*(seg<=g.last)

def extract_seg(segmentation,g):
    seg = segmentation[:,g.bbox[1]]
    return seg*(seg>=g.first)*(seg<=g.last)

def all_gaps(c):
    """Computes a list of all the pairwise gaps between
    connected components of an image."""
    # c,n = morph.label(c>0)
    n = len(unique(c))-1
    if n<2: return []
    #imshow(where(c>0,c%3+1,0),cmap=cm.jet)
    dts = []
    for i,v in enumerate(unique(c)):
        if v==0: continue
        dt = morphology.distance_transform_edt(c!=v)
        dts.append(dt)
    result = []
    for i in range(len(dts)):
        for j in range(i+1,len(dts)):
            dt1,dt2 = dts[i],dts[j]
            dtm = minimum(dt1,dt2)
            mv = amin(dtm[(dt1<=dtm+1)&(dt2<=dtm+1)])
            result.append((i,j,mv))
            result.append((j,i,mv))
    return result

def all_min_gaps(c):
    """For each connected component, computes the minimum distance
    of that component from some neighboring component.  For
    character segmentation, we usually want to have few gaps
    and small maximum distances."""
    pgaps = all_gaps(c)
    if len(pgaps)==0: return []
    components = unique([x[0] for x in pgaps])
    return [amin([d for i,j,d in pgaps if i==c]) for c in components]

def non_noise_components(seg,threshold=0.1):
    """Estimate the number of non-noise connected components in a character
    image. This computes the size of all connected components, and it considers
    all components of size less then `threshold` times the size of the largest
    component to be noise."""
    seg = 1*(seg>0)
    labels,n = morph.label(seg)
    totals = measurements.sum(seg,labels,range(1,n+1))
    return sum(totals>amax(totals)*threshold)

def has_limited_gaps(segmentation,group,maxcomp=2,maxgapsize=2):
    """Given a segmentation and a group, a triple (first,last,bbox), 
    determine whether it is a "good group", i.e., whether it has 
    no more than `maxcomp` connected components and no gaps
    larger than `maxgapsize`."""
    seg = extract_seg(segmentation,group)
    if non_noise_components(seg)>maxcomp: return 0
    gaps = all_min_gaps(seg)
    if len(gaps)==0: return 1
    if amax(gaps)>maxgapsize: return 0
    return 1

def number_of_vertical_strokes(seg,debug=0):
    if debug: imshow(seg>0)
    proj = sum(seg>0,axis=0)
    proj = filters.gaussian_filter(1.0*proj,1.0,mode='constant')
    if debug: plot(proj,'b')
    peaks = (proj>=filters.maximum_filter(proj,3))
    peaks *= (proj>=0.5*amax(proj))
    if debug: plot(peaks*amax(proj),'r')
    return sum(peaks)

def number_of_holes(seg):
    holes = morphology.binary_fill_holes(seg)-seg
    _,n = morph.label(holes)
    return n

def good_complexity(segmentation,group,maxpeaks=3,maxholes=3):
    """Given a segmentation and a group, determines whether
    the segment has a "good complexity", that is, not too many
    holes and not too many vertical strokes."""
    seg = extract_seg(segmentation,group)
    seg = (seg>0)
    npeaks = number_of_vertical_strokes(seg)
    if npeaks>maxpeaks: return 0
    nholes = number_of_holes(seg)
    if nholes>maxholes: return 0
    return 1

def extract_rsegs(segmentation,min_aspect=0.1,max_aspect=1.8,maxrange=3,maxcomp=4,
                  maxgapsize=2,maxpeaks=4,maxholes=3):
    """Given a segmentation, extracts a list of Segment objects representing
    character candidates.
    
    Character candidates are filtered based on:

    - `min_aspect`: minimum aspect ratio
    - `max_aspect`: maximum aspect ratio
    - `maxrange`: maximum # segments to group together
    - `maxcomp`: maximum # connected components in a character
    - `maxgapsize`: maximum gap between segments within a component
    - `maxpeaks`: maximum number of estimated vertical strokes
    - `maxholes`: maximum number of estimated holes

    The default parameters are set for printed Latin characters of reasonable
    quality and with not too many diacritics.
    """
    groups = extract_candidate_groups(segmentation,
                                      min_aspect=min_aspect,max_aspect=max_aspect,
                                      maxrange=maxrange)
    ggroups = [g for g in groups \
               if has_limited_gaps(segmentation,g,maxcomp=maxcomp,maxgapsize=maxgapsize)]
    ggroups = [g for g in ggroups \
               if good_complexity(segmentation,g,maxpeaks=maxpeaks,maxholes=maxholes)]
    result = [g.replace(img=extract_char(segmentation,g)) for g in ggroups]
    return result

def extract_csegs(segmentation,aligned=None):
    """Given a segmentation, extracts a list of segment objects."""
    if aligned is None: aligned = []
    def get(c): return aligned[c] if c<len(aligned) else ""
    boxes = morph.find_objects(segmentation)
    n = len(boxes)
    result = [Segment(first=i+1,last=i+1,bbox=box,out=[(get(i),0.0)],
                      img=1*(segmentation[:,box[1]]==i+1)) \
                      for i,box in enumerate(boxes) if box is not None]
    return result

###
### Finding missegmented characters
###

import improc
from scipy.ndimage import interpolation,filters
from scipy import signal

def best_correlation(u,l,pad=1):
    """Given an image `u` and a list of images `l`, finds the image
    with the best normalized cross correlation in `l`.  Correlation
    is carried out with `mode="valid"` after padding with size pad,
    and only for images whose dimensions are within `pad` of each other."""
    def normalize(u):
        u = u-mean(u)
        return u/norm(u)
    u = normalize(u)
    padded = improc.pad_by(u,pad)
    normalized = []
    for v in l:
        if amax(abs(array(u.shape)-array(v.shape)))<=pad:
            normalized.append(normalize(v))
    convs = []
    for v in normalized:
        convs.append(signal.correlate(padded,v,mode='valid'))
    global _convs
    _convs += sum([c.size for c in convs])
    return amax([amax(c) for c in convs]) if len(convs)>0 else 0.0

def extract_non_csegs(rsegs,csegs,threshold=0.95,pad=1):
    """Given a list of raw segmentations and character segmentations
    (each of the form `[(image,i,j,box),...]`), find all the characters
    in `rsegs` that have a correlation of less than threshold
    with some character in `csegs`."""
    global _convs
    _convs = 0
    cimgs = [c.img for c in csegs]
    result = []
    for r in rsegs:
        best = best_correlation(r.img,cimgs,pad=pad)
        if best>threshold: continue
        result.append(r)
    return result

min_xheight = 10
max_xheight = 30

def check_line_image(image):
    if image.shape[0]<10:
        raise BadImage("line image not high enough (maybe rescale?)",image=image)
    if image.shape[0]>200:
        raise BadImage("line image too high (maybe rescale?)",image=image)
    if image.shape[1]<10:
        raise BadImage("line image not wide enough (segmentation error?)",image=image)
    if image.shape[1]>10000:
        raise BadImage("line image too wide???",image=image)
    if mean(image)<0.5*amax(image):
        raise BadImage("image may not be white on black text (maybe invert?)",image=image)
    if sum(image)<20:
        raise BadImage("not enough pixels in image, maybe the line is empty?",image=image)
    xheight,_ = lineproc.estimate_xheight(1-image)
    if xheight<min_xheight:
        raise BadImage("xheight %f too small (maybe rescale?)"%xheight,image=image)
    if xheight>max_xheight:
        raise BadImage("xheight %f too large (maybe rescale?)"%xheight,image=image)

def clean_line_image(image,latin=1):
    # convert to float, normalize
    image = image*1.0/amax(image)
    # optional cleanup
    if latin:
        image = 1-image
        image = lineproc.latin_filter(image,r=self.latin_r)
        image = lineproc.remove_noise(image,self.noise_threshold)
        image = 1-image
    return image

def read_lattice(fname):
    segments = []
    with open(fname) as stream:
        for line in stream.readlines():
            if line[0]=="#": continue
            f = line.split()
            if f[0]=="segment":
                first,last = [int(x) for x in f[2].split(":")]
                b = [int(x) for x in f[3].split(":")]
                bbox = (slice(b[0],b[1]),slice(b[2],b[3]))
                sp = [float(x) for x in f[4:6]]
                segments.append(Segment(first=first,last=last,bbox=bbox,sp=sp,out=[]))
            elif f[0]=="chr":
                segments[-1].out.append((float(f[2]),f[3]))
            else:
                raise Internal("unknown start of line: "+line)
    return segments

def write_lattice(stream,segments):
    segments = sorted(segments,key=lambda s:(s.first,s.last))
    for i,s in enumerate(segments):
        b = s.bbox
        b = (b[0].start,b[0].stop,b[1].start,b[1].stop)
        stream.write("segment %d\t%d:%d\t%d:%d:%d:%d\t%.2f\t%.2f\n"%((i,s.first,s.last)+b+tuple(s.sp)))
        for j,(cls,cost) in enumerate(s.out):
            stream.write("chr %d\t%d\t%.4f\t%s\n"%(i,j,cost,cls))

import heapq

def shortest_path(transitions,start=0,end=None,noreject=1):
    """Simple implementation of shortest path algorithm, used
    for finding the language model free best result from the
    grouper.  For more complex searches, use the FST library 
    and a language model."""
    if end is None: end = len(transitions)-1
    # each transition is (cost,dest,label)
    costs = [inf]*len(transitions)
    sources = [None]*len(transitions)
    labels = [""]*len(transitions)
    costs[start] = 0.0
    queue = []
    heapq.heappush(queue,(0.0,start))
    while len(queue)>0:
        (ocost,i) = heapq.heappop(queue)
        for (cost,j,label) in transitions[i]:
            if noreject and "~" in label: continue
            ncost = ocost+cost
            if ncost<costs[j]:
                sources[j] = i
                costs[j] = ncost
                labels[j] = label
                heapq.heappush(queue,(costs[j],j))
    if costs[end]==inf:
        return None,None,None
    rcosts = []
    rstates = []
    result = []
    rtrans = []
    j = end
    while j!=start:
        rcosts.append(costs[j])
        result.append(labels[j])
        rstates.append(j)
        rtrans.append((sources[j],j))
        j = sources[j]
    return list(reversed(result)),list(reversed(rcosts)),list(reversed(rtrans))

def bestpath(segs,**kw):
    """Compute a best path through the lattice for debugging purposes.
    This assumes that all costs are non-negative and that there are no loops."""
    n = 1+max([s.last for s in segs])
    transitions = [[] for i in range(n)]
    for i,s in enumerate(segs):
        sp = " " if s.sp[0]<s.sp[1] else ""
        for cls,cost in s.out:
            transitions[s.first-1].append((cost,s.last,cls+sp))
    #for k,v in enumerate(transitions): print k,v
    labels,costs,trans = shortest_path(transitions,0,n-1,**kw)
    if labels is None: return None,None,None
    trans = [(i+1,j) for i,j in trans]
    return labels,costs,trans

