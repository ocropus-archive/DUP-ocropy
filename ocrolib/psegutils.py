import pdb
from pylab import *
import argparse,glob,os,os.path
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
from scipy.misc import imsave

class record:
    def __init__(self,**kw): self.__dict__.update(kw)

class B_:
    def __or__(self,other):
        if other.dtype==dtype('B'): return other
        return array(other,'B')
B = B_()

def check_binary(image):
    assert image.dtype=='B' or image.dtype=='i' or image.dtype==dtype('bool'),\
        "array should be binary, is %s %s"%(image.dtype,image.shape)
    assert amin(image)>=0 and amax(image)<=1,\
        "array should be binary, has values %g to %g"%(amin(image),amax(image))
def r_dilation(image,size,origin=0):
    check_binary(image)
    return filters.maximum_filter(image,size,origin=origin)
def r_erosion(image,size,origin=0):
    check_binary(image)
    return filters.minimum_filter(image,size,origin=origin)
def r_opening(image,size,origin=0):
    check_binary(image)
    image = r_erosion(image,size,origin=origin)
    return r_dilation(image,size,origin=origin)
def r_closing(image,size,origin=0):
    check_binary(image)
    image = r_dilation(image,size,origin=0)
    return r_erosion(image,size,origin=0)

def rb_dilation(image,size,origin=0):
    output = zeros(image.shape,'f')
    filters.uniform_filter(image,size,output=output,origin=origin,mode='constant',cval=0)
    return (output>0)
def rb_erosion(image,size,origin=0):
    output = zeros(image.shape,'f')
    filters.uniform_filter(image,size,output=output,origin=origin,mode='constant',cval=1)
    return (output==1)
def rb_opening(image,size,origin=0):
    check_binary(image)
    image = rb_erosion(image,size,origin=origin)
    return rb_dilation(image,size,origin=origin)
def rb_closing(image,size,origin=0):
    check_binary(image)
    image = rb_dilation(image,size,origin=origin)
    return rb_erosion(image,size,origin=origin)

def rg_dilation(image,size,origin=0):
    return filters.maximum_filter(image,size,origin=origin)
def rg_erosion(image,size,origin=0):
    return filters.minimum_filter(image,size,origin=origin)
def rg_opening(image,size,origin=0):
    image = r_erosion(image,size,origin=origin)
    return r_dilation(image,size,origin=origin)
def rg_closing(image,size,origin=0):
    image = r_dilation(image,size,origin=0)
    return r_erosion(image,size,origin=0)

def progress(*args):
    if len(args)==0: print; return
    l = ["%s"%l for l in args]
    sys.stdout.write(" ".join(l)+" ")
    sys.stdout.flush()

def segshow(x):
    imshow(where(x==0,0,10+100*sin(10*x)**2),cmap=cm.cubehelix)

def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,return_distances=1,return_indices=1)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread

def keep_marked(image,markers):
    labels,_ = measurements.label(image)
    marked = unique(labels*(markers!=0))
    kept = in1d(labels.ravel(),marked)
    return kept.reshape(*labels.shape)

def remove_marked(image,markers):
    marked = keep_marked(image,markers)
    return image*(marked==0)

def correspondences(labels1,labels2):
    q = 100000
    assert amin(labels1)>=0 and amin(labels2)>=0
    assert amax(labels2)<q
    combo = labels1*q+labels2
    result = unique(combo)
    result = array([result//q,result%q])
    return result

def propagate_labels_simple(regions,labels):
    """Spread the labels to the corresponding regions."""
    rlabels,_ = measurements.label(regions)
    cors = correspondences(rlabels,labels)
    outputs = zeros(amax(rlabels)+1,'i')
    for o,i in cors.T: outputs[o] = i
    outputs[0] = 0
    return outputs[rlabels]

def propagate_labels(regions,labels,conflict=0):
    """Spread the labels to the corresponding regions."""
    rlabels,_ = measurements.label(regions)
    cors = correspondences(rlabels,labels)
    outputs = zeros(amax(rlabels)+1,'i')
    oops = -(1<<30)
    for o,i in cors.T:
        if outputs[o]!=0: outputs[o] = oops
        else: outputs[o] = i
    outputs[outputs==oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]

def H(s): return s[0].stop-s[0].start
def W(s): return s[1].stop-s[1].start
def A(s): return W(s)*H(s)
def M(s): return mean([s[0].start,s[0].stop]),mean([s[1].start,s[1].stop])

def binary_objects(binary):
    labels,n = measurements.label(binary)
    objects = measurements.find_objects(labels)
    return objects

def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=A)
    scalemap = zeros(binary.shape)
    for o in bysize:
        if amax(scalemap[o])>0: continue
        scalemap[o] = A(o)**0.5
    scale = median(scalemap[(scalemap>3)&(scalemap<100)])
    return scale

def compute_boxmap(binary,scale,threshold=(.5,4),dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=A)
    boxmap = zeros(binary.shape,dtype)
    for o in bysize:
        if A(o)**.5<threshold[0]*scale: continue
        if A(o)**.5>threshold[1]*scale: continue
        boxmap[o] = 1
    return boxmap

def compute_lines(segmentation,scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = measurements.find_objects(segmentation)
    lines = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        if W(o)<2*scale or H(o)<scale: continue
        mask = (segmentation[o]==i+1)
        if amax(mask)==0: continue
        result = record()
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines

def pad_image(image,d,cval=inf):
    result = ones(array(image.shape)+2*d)
    result[:,:] = amax(image) if cval==inf else cval
    result[d:-d,d:-d] = image
    return result

def extract(image,y0,x0,y1,x1,mode='nearest',cval=0):
    h,w = image.shape
    ch,cw = y1-y0,x1-x0
    y,x = clip(y0,0,h-ch),clip(x0,0,w-cw)
    sub = image[y:y+ch,x:x+cw]
    return interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)

def extract_masked(image,l,pad=5,expand=0):
    y0,x0,y1,x1 = l.bounds[0].start,l.bounds[1].start,l.bounds[0].stop,l.bounds[1].stop
    if pad>0:
        mask = pad_image(l.mask,pad,cval=0)
    else:
        mask = l.mask
    line = extract(image,y0-pad,x0-pad,y1+pad,x1+pad)
    if expand>0:
        mask = filters.maximum_filter(mask,(expand,expand))
    line = where(mask,line,amax(line))
    return line

def reading_order(lines,highlight=None,debug=0):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = zeros((len(lines),len(lines)),'B')
    def x_overlaps(u,v):
        return u[1].start<v[1].stop and u[1].stop>v[1].start
    def above(u,v):
        return u[0].start<v[0].start
    def left_of(u,v):
        return u[1].stop<v[1].start
    def separates(w,u,v):
        if w[0].stop<min(u[0].start,v[0].start): return 0
        if w[0].start>max(u[0].stop,v[0].stop): return 0
        if w[1].start<u[1].stop and w[1].stop>v[1].start: return 1
    if highlight is not None:
        clf(); title("highlight"); imshow(binary); ginput(1,debug)
    for i,u in enumerate(lines):
        for j,v in enumerate(lines):
            if x_overlaps(u,v):
                if above(u,v):
                    order[i,j] = 1
            else:
                if [w for w in lines if separates(w,u,v)]==[]:
                    if left_of(u,v): order[i,j] = 1
            if j==highlight and order[i,j]:
                print (i,j),
                y0,x0 = M(lines[i])
                y1,x1 = M(lines[j])
                plot([x0,x1+200],[y0,y1])
    if highlight is not None:
        print
        ginput(1,debug)
    return order

def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = zeros(n)
    L = []
    def visit(k):
        if visited[k]: return
        visited[k] = 1
        for l in find(order[:,k]):
            visit(l)
        L.append(k)
    for k in range(n):
        visit(k)
    return L #[::-1]

def show_lines(image,lines,lsort):
    """Overlays the computed lines on top of the image, for debugging
    purposes."""
    ys,xs = [],[]
    clf(); cla()
    imshow(image)
    for i in range(len(lines)):
        l = lines[lsort[i]]
        y,x = M(l.bounds)
        xs.append(x)
        ys.append(y)
        o = l.bounds
        r = matplotlib.patches.Rectangle((o[1].start,o[0].start),edgecolor='r',fill=0,width=W(o),height=H(o))
        gca().add_patch(r)
    h,w = image.shape
    ylim(h,0); xlim(0,w)
    plot(xs,ys)

def read_gray(fname):
    image = imread(fname)
    if image.ndim==3: image = mean(image,2)
    return image

def read_binary(fname):
    image = imread(fname)
    if image.ndim==3: image = mean(image,2)
    image -= amin(image)
    image /= amax(image)
    assert sum(image<0.01)+sum(image>0.99)>0.99*prod(image.shape),"input image is not binary"
    binary = 1.0*(image<0.5)
    return binary

def rgbshow(r,g,b=None,gn=1,cn=0,ab=0,**kw):
    """Small function to display 2 or 3 images as RGB channels."""
    if b is None: b = zeros(r.shape)
    combo = transpose(array([r,g,b]),axes=[1,2,0])
    if cn:
        for i in range(3):
            combo[:,:,i] /= max(abs(amin(combo[:,:,i])),abs(amax(combo[:,:,i])))
    elif gn:
        combo /= max(abs(amin(combo)),abs(amax(combo)))
    if ab:
        combo = abs(combo)
    if amin(combo)<0: print "warning: values less than zero"
    imshow(clip(combo,0,1),**kw)

def select_regions(binary,f,min=0,nbest=100000):
    labels,n = measurements.label(binary)
    objects = measurements.find_objects(labels)
    scores = [f(o) for o in objects]
    best = argsort(scores)
    keep = zeros(len(objects)+1,'B')
    for i in best[-nbest:]:
        if scores[i]<=min: continue
        keep[i+1] = 1
    return keep[labels]

def all_neighbors(image):
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

def norm_max(v):
    return v/amax(v)

def compute_separators_morph(binary,scale):
    thick = r_dilation(binary,(max(5,scale/4),max(5,scale)))
    vert = rb_opening(thick,(10*scale,1))
    vert = select_regions(vert,W,min=3,nbest=5)
    vert = select_regions(vert,H,min=20*scale,nbest=3)
    return vert

def compute_columns_morph(binary,scale,debug=0,maxcols=3,minheight=20,maxwidth=5):
    boxmap = compute_boxmap(binary,scale,dtype='B')
    bounds = rb_closing(B|boxmap,(5*scale,5*scale)) 
    if debug>0:
        clf(); title("bounds"); imshow(0.3*boxmap+0.7*bounds); ginput(1,debug)
    bounds = maximum(B|1-bounds,B|boxmap)
    if debug>0:
        clf(); title("input"); imshow(0.3*boxmap+0.7*bounds); ginput(1,debug)
    cols = 1-rb_closing(boxmap,(20*scale,scale))
    if debug>0:
        clf(); title("columns0"); imshow(0.3*boxmap+0.7*cols); ginput(1,debug)
    cols = select_regions(cols,lambda x:-W(x),min=-maxwidth*scale)
    if debug>0:
        clf(); title("columns1"); imshow(0.3*boxmap+0.7*cols); ginput(1,debug)
    cols = select_regions(cols,H,min=minheight*scale,nbest=maxcols)
    if debug>0:
        clf(); title("columns2"); imshow(0.3*boxmap+0.7*cols); ginput(1,debug)
    cols = r_erosion(cols,(scale,0))
    cols = r_dilation(cols,(scale,0),origin=(int(scale/2)-1,0))
    if debug>0:
        clf(); title("columns3"); imshow(0.3*boxmap+0.7*cols); ginput(1,debug)
    return cols

