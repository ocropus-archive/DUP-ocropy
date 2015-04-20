from toplevel import *
from pylab import *
from scipy.ndimage import filters,interpolation
import sl,morph

def B(a):
    if a.dtype==dtype('B'): return a
    return array(a,'B')

class record:
    def __init__(self,**kw): self.__dict__.update(kw)

def blackout_images(image,ticlass):
    """Takes a page image and a ticlass text/image classification image and replaces
    all regions tagged as 'image' with rectangles in the page image.  The page image
    is modified in place.  All images are iulib arrays."""
    rgb = ocropy.intarray()
    ticlass.textImageProbabilities(rgb,image)
    r = ocropy.bytearray()
    g = ocropy.bytearray()
    b = ocropy.bytearray()
    ocropy.unpack_rgb(r,g,b,rgb)
    components = ocropy.intarray()
    components.copy(g)
    n = ocropy.label_components(components)
    print "[note] number of image regions",n
    tirects = ocropy.rectarray()
    ocropy.bounding_boxes(tirects,components)
    for i in range(1,tirects.length()):
        r = tirects.at(i)
        ocropy.fill_rect(image,r,0)
        r.pad_by(-5,-5)
        ocropy.fill_rect(image,r,255)
        
def binary_objects(binary):
    labels,n = morph.label(binary)
    objects = morph.find_objects(labels)
    return objects

def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=sl.area)
    scalemap = zeros(binary.shape)
    for o in bysize:
        if amax(scalemap[o])>0: continue
        scalemap[o] = sl.area(o)**0.5
    scale = median(scalemap[(scalemap>3)&(scalemap<100)])
    return scale

def compute_boxmap(binary,scale,threshold=(.5,4),dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=sl.area)
    boxmap = zeros(binary.shape,dtype)
    for o in bysize:
        if sl.area(o)**.5<threshold[0]*scale: continue
        if sl.area(o)**.5>threshold[1]*scale: continue
        boxmap[o] = 1
    return boxmap

def compute_lines(segmentation,scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        if sl.dim1(o)<2*scale or sl.dim0(o)<scale: continue
        mask = (segmentation[o]==i+1)
        if amax(mask)==0: continue
        result = record()
        result.label = i+1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines

def pad_image(image,d,cval=inf):
    result = ones(array(image.shape)+2*d)
    result[:,:] = amax(image) if cval==inf else cval
    result[d:-d,d:-d] = image
    return result

@checks(ARANK(2),int,int,int,int,mode=str,cval=True,_=GRAYSCALE)
def extract(image,y0,x0,y1,x1,mode='nearest',cval=0):
    h,w = image.shape
    ch,cw = y1-y0,x1-x0
    y,x = clip(y0,0,max(h-ch,0)),clip(x0,0,max(w-cw, 0))
    sub = image[y:y+ch,x:x+cw]
    # print "extract",image.dtype,image.shape
    try:
        r = interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
        if cw > w or ch > h:
            pady0, padx0 = max(-y0, 0), max(-x0, 0)
            r = interpolation.affine_transform(r, eye(2), offset=(pady0, padx0), cval=1, output_shape=(ch, cw))
        return r

    except RuntimeError:
        # workaround for platform differences between 32bit and 64bit
        # scipy.ndimage
        dtype = sub.dtype
        sub = array(sub,dtype='float64')
        sub = interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
        sub = array(sub,dtype=dtype)
        return sub

@checks(ARANK(2),True,pad=int,expand=int,_=GRAYSCALE)
def extract_masked(image,linedesc,pad=5,expand=0):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    y0,x0,y1,x1 = [int(x) for x in [linedesc.bounds[0].start,linedesc.bounds[1].start, \
                  linedesc.bounds[0].stop,linedesc.bounds[1].stop]]
    if pad>0:
        mask = pad_image(linedesc.mask,pad,cval=0)
    else:
        mask = linedesc.mask
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
                y0,x0 = sl.center(lines[i])
                y1,x1 = sl.center(lines[j])
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
        y,x = sl.center(l.bounds)
        xs.append(x)
        ys.append(y)
        o = l.bounds
        r = matplotlib.patches.Rectangle((o[1].start,o[0].start),edgecolor='r',fill=0,width=sl.dim1(o),height=sl.dim0(o))
        gca().add_patch(r)
    h,w = image.shape
    ylim(h,0); xlim(0,w)
    plot(xs,ys)

@obsolete
def read_gray(fname):
    image = imread(fname)
    if image.ndim==3: image = mean(image,2)
    return image

@obsolete
def read_binary(fname):
    image = imread(fname)
    if image.ndim==3: image = mean(image,2)
    image -= amin(image)
    image /= amax(image)
    assert sum(image<0.01)+sum(image>0.99)>0.99*prod(image.shape),"input image is not binary"
    binary = 1.0*(image<0.5)
    return binary

@obsolete
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

