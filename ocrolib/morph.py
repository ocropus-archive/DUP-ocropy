################################################################
### various add-ons to the SciPy morphology package
################################################################

# pylint: disable=bad-whitespace
# pylint: disable=bare-except
# pylint: disable=multiple-statements
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import

from __future__ import print_function

# TODO make explicit imports
from numpy import *
import pylab
from pylab import *
from scipy.ndimage import morphology,measurements,filters
from scipy.ndimage.morphology import *
from ocrolib.toplevel import *

@checks(ABINARY2)
def label(image,**kw):
    """Redefine the scipy.ndimage.measurements.label function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms."""
    try: return measurements.label(image,**kw)
    except: pass
    types = ["int32","uint32","int64","uint64","int16","uint16"]
    for t in types:
        try: return measurements.label(array(image,dtype=t),**kw)
        except: pass
    # let it raise the same exception as before
    return measurements.label(image,**kw)

@checks(AINT2)
def find_objects(image,**kw):
    """Redefine the scipy.ndimage.measurements.find_objects function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms."""
    try: return measurements.find_objects(image,**kw)
    except: pass
    types = ["int32","uint32","int64","uint64","int16","uint16"]
    for t in types:
        try: return measurements.find_objects(array(image,dtype=t),**kw)
        except: pass
    # let it raise the same exception as before
    return measurements.find_objects(image,**kw)

def check_binary(image):
    assert image.dtype=='B' or image.dtype=='i' or image.dtype==dtype('bool'),\
        "array should be binary, is %s %s"%(image.dtype,image.shape)
    assert amin(image)>=0 and amax(image)<=1,\
        "array should be binary, has values %g to %g"%(amin(image),amax(image))

@checks(ABINARY2,uintpair)
def r_dilation(image,size,origin=0):
    """Dilation with rectangular structuring element using maximum_filter"""
    return filters.maximum_filter(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def r_erosion(image,size,origin=0):
    """Erosion with rectangular structuring element using maximum_filter"""
    return filters.minimum_filter(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def r_opening(image,size,origin=0):
    """Opening with rectangular structuring element using maximum/minimum filter"""
    check_binary(image)
    image = r_erosion(image,size,origin=origin)
    return r_dilation(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def r_closing(image,size,origin=0):
    """Closing with rectangular structuring element using maximum/minimum filter"""
    check_binary(image)
    image = r_dilation(image,size,origin=0)
    return r_erosion(image,size,origin=0)

@checks(ABINARY2,uintpair)
def rb_dilation(image,size,origin=0):
    """Binary dilation using linear filters."""
    output = zeros(image.shape,'f')
    filters.uniform_filter(image,size,output=output,origin=origin,mode='constant',cval=0)
    return array(output>0,'i')

@checks(ABINARY2,uintpair)
def rb_erosion(image,size,origin=0):
    """Binary erosion using linear filters."""
    output = zeros(image.shape,'f')
    filters.uniform_filter(image,size,output=output,origin=origin,mode='constant',cval=1)
    return array(output==1,'i')

@checks(ABINARY2,uintpair)
def rb_opening(image,size,origin=0):
    """Binary opening using linear filters."""
    image = rb_erosion(image,size,origin=origin)
    return rb_dilation(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def rb_closing(image,size,origin=0):
    """Binary closing using linear filters."""
    image = rb_dilation(image,size,origin=origin)
    return rb_erosion(image,size,origin=origin)

@checks(GRAYSCALE,uintpair)
def rg_dilation(image,size,origin=0):
    """Grayscale dilation with maximum/minimum filters."""
    return filters.maximum_filter(image,size,origin=origin)

@checks(GRAYSCALE,uintpair)
def rg_erosion(image,size,origin=0):
    """Grayscale erosion with maximum/minimum filters."""
    return filters.minimum_filter(image,size,origin=origin)

@checks(GRAYSCALE,uintpair)
def rg_opening(image,size,origin=0):
    """Grayscale opening with maximum/minimum filters."""
    image = r_erosion(image,size,origin=origin)
    return r_dilation(image,size,origin=origin)

@checks(GRAYSCALE,uintpair)
def rg_closing(image,size,origin=0):
    """Grayscale closing with maximum/minimum filters."""
    image = r_dilation(image,size,origin=0)
    return r_erosion(image,size,origin=0)

@checks(SEGMENTATION)
def showlabels(x,n=7):
    pylab.imshow(where(x>0,x%n+1,0),cmap=pylab.cm.gist_stern)

@checks(SEGMENTATION)
def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,return_distances=1,return_indices=1)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread

@checks(ABINARY2,ABINARY2)
def keep_marked(image,markers):
    """Given a marker image, keep only the connected components
    that overlap the markers."""
    labels,_ = label(image)
    marked = unique(labels*(markers!=0))
    kept = in1d(labels.ravel(),marked)
    return (image!=0)*kept.reshape(*labels.shape)

@checks(ABINARY2,ABINARY2)
def remove_marked(image,markers):
    """Given a marker image, remove all the connected components
    that overlap markers."""
    marked = keep_marked(image,markers)
    return image*(marked==0)

@checks(SEGMENTATION,SEGMENTATION)
def correspondences(labels1,labels2):
    """Given two labeled images, compute an array giving the correspondences
    between labels in the two images."""
    q = 100000
    assert amin(labels1)>=0 and amin(labels2)>=0
    assert amax(labels2)<q
    combo = labels1*q+labels2
    result = unique(combo)
    result = array([result//q,result%q])
    return result

@checks(ABINARY2,SEGMENTATION)
def propagate_labels_simple(regions,labels):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label."""
    rlabels,_ = label(regions)
    cors = correspondences(rlabels,labels)
    outputs = zeros(amax(rlabels)+1,'i')
    for o,i in cors.T: outputs[o] = i
    outputs[0] = 0
    return outputs[rlabels]

@checks(ABINARY2,SEGMENTATION)
def propagate_labels(image,labels,conflict=0):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label.
    Assign the value `conflict` to any labels that have a conflict."""
    rlabels,_ = label(image)
    cors = correspondences(rlabels,labels)
    outputs = zeros(amax(rlabels)+1,'i')
    oops = -(1<<30)
    for o,i in cors.T:
        if outputs[o]!=0: outputs[o] = oops
        else: outputs[o] = i
    outputs[outputs==oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]

@checks(ABINARY2,True)
def select_regions(binary,f,min=0,nbest=100000):
    """Given a scoring function f over slice tuples (as returned by
    find_objects), keeps at most nbest regions whose scores is higher
    than min."""
    labels,n = label(binary)
    objects = find_objects(labels)
    scores = [f(o) for o in objects]
    best = argsort(scores)
    keep = zeros(len(objects)+1,'i')
    if nbest > 0:
        for i in best[-nbest:]:
            if scores[i]<=min: continue
            keep[i+1] = 1
    # print scores,best[-nbest:],keep
    # print sorted(list(set(labels.ravel())))
    # print sorted(list(set(keep[labels].ravel())))
    return keep[labels]

@checks(SEGMENTATION)
def all_neighbors(image):
    """Given an image with labels, find all pairs of labels
    that are directly neighboring each other."""
    q = 100000
    assert amax(image)<q
    assert amin(image)>=0
    u = unique(q*image+roll(image,1,0))
    d = unique(q*image+roll(image,-1,0))
    l = unique(q*image+roll(image,1,1))
    r = unique(q*image+roll(image,-1,1))
    all = unique(r_[u,d,l,r])
    all = c_[all//q,all%q]
    all = unique(array([sorted(x) for x in all]))
    return all

################################################################
### Iterate through the regions of a color image.
################################################################

@checks(SEGMENTATION)
def renumber_labels_ordered(a,correspondence=0):
    """Renumber the labels of the input array in numerical order so
    that they are arranged from 1...N"""
    assert amin(a)>=0
    assert amax(a)<=2**25
    labels = sorted(unique(ravel(a)))
    renum = zeros(amax(labels)+1,dtype='i')
    renum[labels] = arange(len(labels),dtype='i')
    if correspondence:
        return renum[a],labels
    else:
        return renum[a]

@checks(SEGMENTATION)
def renumber_labels(a):
    """Alias for renumber_labels_ordered"""
    return renumber_labels_ordered(a)

def fallback_cmp(a, b):
    return (a > b) - (a < b)

def pyargsort(seq,cmp=fallback_cmp, key=lambda x:x):
    """Like numpy's argsort, but using the builtin Python sorting
    function.  Takes an optional cmp."""
    return sorted(range(len(seq)),key=lambda x:key(seq.__getitem__(x)),cmp=cmp)

@checks(SEGMENTATION)
def renumber_by_xcenter(seg):
    """Given a segmentation (as a color image), change the labels
    assigned to each region such that when the labels are considered
    in ascending sequence, the x-centers of their bounding boxes
    are non-decreasing.  This is used for sorting the components
    of a segmented text line into left-to-right reading order."""
    objects = [(slice(0,0),slice(0,0))]+find_objects(seg)
    def xc(o):
        # if some labels of the segmentation are missing, we
        # return a very large xcenter, which will move them all
        # the way to the right (they don't show up in the final
        # segmentation anyway)
        if o is None: return 999999
        return mean((o[1].start,o[1].stop))
    xs = array([xc(o) for o in objects])
    order = argsort(xs)
    segmap = zeros(amax(seg)+1,'i')
    for i,j in enumerate(order): segmap[j] = i
    return segmap[seg]

@checks(SEGMENTATION)
def ordered_by_xcenter(seg):
    """Verify that the labels of a segmentation are ordered
    spatially (as determined by the x-center of their bounding
    boxes) in left-to-right reading order."""
    objects = [(slice(0,0),slice(0,0))]+find_objects(seg)
    def xc(o): return mean((o[1].start,o[1].stop))
    xs = array([xc(o) for o in objects])
    for i in range(1,len(xs)):
        if xs[i-1]>xs[i]: return 0
    return 1
