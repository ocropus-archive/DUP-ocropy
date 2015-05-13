################################################################
### utilities for lists of slices, treating them like rectangles
################################################################

import numpy
from pylab import mean

### inquiry functions

def is_slices(u):
    for s in u:
        if not isinstance(s, slice): return False
    return True
def dims(s):
    """List of dimensions of the slice list."""
    return tuple([x.stop-x.start for x in s])
def dim(s,i):
    """Dimension of the slice list for index i."""
    return s[i].stop-s[i].start
def dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop-s[0].start
def dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop-s[1].start
def raster(u):
    """Return (row0,row1,col0,col1)."""
    return (u[0].start,u[0].stop,u[1].start,u[1].stop)
def box(r0,r1,c0,c1):
    return (slice(r0,r1),slice(c0,c1))
def start(u):
    return tuple([x.start for x in u])
def stop(u):
    return tuple([x.stop for x in u])
def bounds(a):
    """Return a list of slices corresponding to the array bounds."""
    return tuple([slice(0,a.shape[i]) for i in range(a.ndim)])
def volume(a):
    """Return the area of the slice list."""
    return numpy.prod([max(x.stop-x.start,0) for x in a])
def empty(a):
    """Test whether the slice is empty."""
    return a is None or volume(a)==0
def shift(u,offsets,scale=1):
    u = list(u)
    for i in range(len(offsets)):
        u[i] = slice(u[i].start+scale*offsets[i],u[i].stop+scale*offsets[i])
    return tuple(u)

### These are special because they only operate on the first two
### dimensions.  That's useful for RGB images.

def area(a):
    """Return the area of the slice list (ignores anything past a[:2]."""
    return numpy.prod([max(x.stop-x.start,0) for x in a[:2]])
def aspect(a):
    return height(a)*1.0/width(a)

### Geometric operations.

def pad(u,d):
    """Pad the slice list by the given amount."""
    return tuple([slice(u[i].start-d,u[i].stop+d) for i in range(len(u))])
def union(u,v):
    """Compute the union of the two slice lists."""
    if u is None: return v
    if v is None: return u
    return tuple([slice(min(u[i].start,v[i].start),max(u[i].stop,v[i].stop)) for i in range(len(u))])
def intersect(u,v):
    """Compute the intersection of the two slice lists."""
    if u is None: return v
    if v is None: return u
    return tuple([slice(max(u[i].start,v[i].start),min(u[i].stop,v[i].stop)) for i in range(len(u))])

def xoverlap(u,v):
    return max(0,min(u[1].stop,v[1].stop)-max(u[1].start,v[1].start))
def yoverlap(u,v):
    return max(0,min(u[0].stop,v[0].stop)-max(u[0].start,v[0].start))
def xoverlap_rel(u,v):
    return xoverlap(u,v)*1.0/max(1,width(u),width(v))
def yoverlap_rel(u,v):
    return yoverlap(u,v)*1.0/max(1,height(u),height(v))
def xoverlaps(u,v):
    return u[1].stop>=v[1].start and v[1].stop>=u[1].start
def yoverlaps(u,v):
    return u[0].stop>=v[0].start and v[0].stop>=u[0].start
    
def center1(s):
    return mean([s[1].stop,s[1].start])
def center0(s):
    return mean([s[0].stop,s[0].start])

def xcenter(s):
    return mean([s[1].stop,s[1].start])
def ycenter(s):
    return mean([s[0].stop,s[0].start])
def center(s):
    return (ycenter(s),xcenter(s))
def center_in(u,v):
    y,x = ycenter(u),xcenter(u)
    return y>=v[0].start and y<=v[0].stop and x>=v[1].start and x<=v[1].stop

def width(s):
    return s[1].stop-s[1].start
def height(s):
    return s[0].stop-s[0].start

### Functions with mathematical coordinate conventions

def mbox(x0,y0,x1,y1,h):
    return (slice(h-y1-1,h-y0-1),slice(x0,x1))
def raster_FIXME(u):
    """Return (y0,x0,y1,x1) for the given height."""
    return (u[0].start,u[1].start,u[0].stop,u[1].stop)
def math(u,h):
    """Return (x0,y0,x1,y1) for the given height."""
    return (u[1].start,h-u[0].stop-1,u[1].stop,h-u[0].start-1)

### Image-related

def extend_to(slices,image):
    if image.ndim==len(slices):
        return slices
    slices = list(slices)
    slices = slices + bounds(image)[len(slices):]
    return tuple(slices)

def cut(image,box,margin=0,bg=0,dtype=None):
    """Cut out a region given by a box (row0,col0,row1,col1),
    with an optional margin."""
    assert len(box)==2 and is_slices(box)
    if dtype is None: dtype = image.dtype
    if image.ndim==3:
        assert image.shape[2]==3
        result = [cut(image[:,:,i],box,margin,bg,dtype) for i in range(image.shape[2])]
        result = numpy.transpose(result,[1,2,0])
        return result
    elif image.ndim==2:
        box = pad(box,margin)
        cbox = intersect(box,bounds(image))
        if empty(cbox):
            result = numpy.empty(dims(box),dtype=dtype)
            result.ravel()[:] = bg
            return result
        cimage = image[cbox]
        if cbox==box:
            return cimage
        else:
            if dtype is None: dtype = image.dtype
            result = numpy.empty(dims(box),dtype=dtype)
            result.ravel()[:] = bg
            moved = shift(cbox,start(box),-1)
            result[moved] = cimage
            return result
    else:
        raise Exception("not implemented for ndim!=2 or ndim!=3")
