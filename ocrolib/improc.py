################################################################
### generic image processing utilities
################################################################

import sys,os,re,glob,math,glob,signal
from numpy import *
from scipy.ndimage import interpolation
import sl
from pylab import *
import scipy
from scipy import stats
from scipy.ndimage import measurements
from pylab import *
from common import *
import common
#from common import method,deprecated
import morph
from numpy import array
from toplevel import *

def deprecated(f):
    return lambda x: x

################################################################
### helper functions
################################################################

def shaped(n):
    if type(n)==int:
        return (n,n)
    elif type(n)==tuple:
        assert type(n[0])==int and type(n[1])==int and len(n)==2
        return n
    else:
        assert False,"%s: not a pair of integers or an integer"

def norm_max(v):
    return v/amax(v)

################################################################
### various functions for cutting out parts of images, padding
### them, etc.
################################################################

def pad_to(image,w,h):
    """Symmetrically pad the image to the given width and height."""
    iw,ih = image.shape
    wd = int(w-iw)
    assert wd>=0
    w0 = wd/2
    w1 = wd-w0
    hd = int(h-ih)
    assert hd>=0
    h0 = hd/2
    h1 = hd-h0
    result = zeros((w,h))
    result[w0:w0+iw,h0:h0+ih] = image
    return result

def pad_by(image,r,dtype=None):
    """Symmetrically pad the image by the given amount"""
    if dtype is None: dtype = image.dtype
    w,h = image.shape
    result = zeros((w+2*r,h+2*r))
    result[r:(w+r),r:(h+r)] = image
    return result

def extract_centered(image,shape,center):
    """Extracts a patch of size `shape` centered on `center`."""
    center = array(center)-array(shape)/2.0
    return interpolation.affine_transform(1.0*image,diag([1,1]),offset=center,
                                          output_shape=shape,order=1)

def extract_centered_scaled(image,shape,center,scale):
    """Extracts a patch of size `shape` centered on `center`
    and scaled by `scale`."""
    scale = 1.0/scale
    rshape = scale*array(shape)
    center = array(center)-rshape/2.0
    result = interpolation.affine_transform(1.0*image,diag([scale,scale]),offset=center,
                                            output_shape=shape,order=1)
    return result

def extract_centered_scaled_barred(image,shape,center,scale,bar=None):
    """Extracts a patch of size `shape` centered on `center`
    and scaled by `scale`. Optionally adds a "bar" to the left side
    of the image, usually used to indicate the baseline and x-line
    of a text line."""
    scale = 1.0/scale
    rshape = scale*array(shape)
    center = array(center)-rshape/2.0
    result = interpolation.affine_transform(1.0*image,diag([scale,scale]),offset=center,
                                            output_shape=shape,order=1)
    if bar is not None:
        bar = array(bar,'f')
        bar -= center[0]
        bar /= scale
        result[int(bar[0]):int(bar[1]),0] = amax(result)

    return result

@deprecated
def cut(image,box,margin=0,bg=0,dtype=None):
    """Cuts out subimages with margins and background.  The box
    is given as (row0,column0,row1,column1).  This uses
    sl.cut (but with different conventions for the box)."""
    (r0,c0,r1,c1) = box
    return sl.cut(image,sl.box(r0,r1,c0,c1),margin=margin,bg=bg,dtype=dtype)

@deprecated
def cut_inefficient(image,box,margin=0,bg=0,dtype=None):
    """Cut out a region given by a box (row0,col0,row1,col1),
    with an optional margin."""
    (r0,c0,r1,c1) = box
    r0 -= margin; c0 -= margin; r1 += margin; c1 += margin
    if dtype is None: dtype = image.dtype
    result = interpolation.shift(image,(-r0,-c0),output=dtype,order=0,cval=bg)
    return result[:(r1-r0),:(c1-c0)]

@deprecated
def pad_bin(char,r=10):
    """Pad to the next bin size."""
    w,h = char.shape
    w = r*int((w+r-1)/r)
    h = r*int((h+r-1)/r)
    return pad_to(char,w,h)

@deprecated
def square(image):
    """Make a numpy array square."""
    w,h = image.shape
    r = max(w,h)
    output = zeros((r,r),image.dtype)
    dx = (r-w)/2
    dy = (r-h)/2
    output[dx:dx+w,dy:dy+h] = image
    return output

@deprecated
def stdsize(image,r=30):
    """Make a numpy array a standard square size."""
    image = square(image)
    s,_ = image.shape
    return interpolation.zoom(image,(r+0.5)/float(s))

@deprecated
def center_maxsize(image,r):
    """Center the image and fit it into an r x r output image.
    If the input is larger in any dimension than r, it is
    scaled down."""
    from pylab import amin,amax,array,zeros
    assert amin(image)>=0 and amax(image)<=1
    image = array(image,'f')
    w,h = image.shape
    s = max(w,h)
    # zoom down, but don't zoom up
    if s>r:
        image = interpolation.zoom(image,(r+0.5)/float(s))
        image[image<0] = 0
        image[image>1] = 1
        w,h = image.shape
    output = zeros((r,r),image.dtype)
    dx = (r-w)/2
    dy = (r-h)/2
    output[dx:dx+w,dy:dy+h] = image
    return output

################################################################
### simple shape comparisons
################################################################

def make_mask(image,r):    
    skeleton = thin(image)
    mask = ~(morphology.binary_dilation(image,iterations=r) - morphology.binary_erosion(image,iterations=r))
    mask |= skeleton # binary_dilation(skeleton,iterations=1)
    return mask

def dist(image,item):
    assert image.shape==item.shape,[image.shape,item.shape]
    ix,iy = measurements.center_of_mass(image)
    if isnan(ix) or isnan(iy): return 9999,9999,None
    # item = (item>amax(item)/2) # in case it's grayscale
    x,y = measurements.center_of_mass(item)
    if isnan(x) or isnan(y): return 9999,9999,None
    dx,dy = int(0.5+x-ix),int(0.5+y-iy)
    shifted = interpolation.shift(image,(dy,dx))
    if abs(dx)>2 or abs(dy)>2:
        return 9999,9999,None
    if 0:
        cla()
        subplot(121); imshow(image-item)
        subplot(122); imshow(shifted-item)
        show()
    image = shifted
    mask = make_mask(image>0.5,1)
    err = sum(mask*abs(item-image))
    total = min(sum(mask*item),sum(mask*image))
    rerr = err/max(1.0,total)
    return err,rerr,image

def symdist(image,item):
    assert type(image)==numpy.ndarray
    assert len(image.shape)==2
    assert len(item.shape)==2
    err,rerr,transformed = dist(image,item)
    err1,rerr1,transformed1 = dist(item,image)
    if rerr<rerr1: return err,rerr,transformed
    else: return err1,rerr1,transformed1


def bbox(image):
    """Compute the bounding box for the pixels in the image."""
    assert len(image.shape)==2,"wrong shape: "+str(image.shape)
    image = array(image!=0,'uint32')
    cs = morph.find_objects(image)
    if len(cs)<1: return None
    c = cs[0]
    return (c[0].start,c[1].start,c[0].stop,c[1].stop)

def extract(image,bbox):
    """Extract a subregion of the given image.  The limits do not have to
    be within the image."""
    r0,c0,r1,c1 = bbox
    assert r0<=r1 and c0<=c1,"%s"%(bbox,)
    result = scipy.ndimage.interpolation.affine_transform(image,diag([1,1]),
                                                          offset=(r0,c0),
                                                          output_shape=(r1-r0,c1-c0))
    assert result.shape == (r1-r0,c1-c0),"docproc.extract failed: %s != %s"%(result.shape,(r1-r0,c1-c0))
    return result

def isotropic_rescale(image,r=32):
    """Rescale the image such that the non-zero pixels fall within a box of size
    r x r.  Rescaling is isotropic."""
    x0,y0,x1,y1 = bbox(image)
    sx = r*1.0/(x1-x0)
    sy = r*1.0/(y1-y0)
    s = min(sx,sy)
    s = min(s,1.0)
    rs = r/s
    dx = x0-(rs-(x1-x0))/2
    dy = y0-(rs-(y1-y0))/2
    result = scipy.ndimage.affine_transform(image,
                                            diag([1/s,1/s]),
                                            offset=(dx,dy),
                                            order=0,
                                            output_shape=(r,r))
    return result

@checks(PATCH,f=float)
def csnormalize(image,f=0.75):
    """Center and size-normalize an image."""
    bimage = 1*(image>mean([amax(image),amin(image)]))
    w,h = bimage.shape
    [xs,ys] = mgrid[0:w,0:h]
    s = sum(bimage)
    if s<1e-4: return image
    s = 1.0/s
    cx = sum(xs*bimage)*s
    cy = sum(ys*bimage)*s
    sxx = sum((xs-cx)**2*bimage)*s
    sxy = sum((xs-cx)*(ys-cy)*bimage)*s
    syy = sum((ys-cy)**2*bimage)*s
    w,v = eigh(array([[sxx,sxy],[sxy,syy]]))
    l = sqrt(amax(w))
    if l>0.01:
        scale = f*max(image.shape)/(4.0*l)
    else:
        scale = 1.0
    m = array([[1.0/scale,0],[0.0,1.0/scale]])
    w,h = image.shape
    c = array([cx,cy])
    d = c-dot(m,array([w/2,h/2]))
    image = interpolation.affine_transform(image,m,offset=d,order=1)
    return image

@checks(PATCH,size=int)
def classifier_normalize(image,size=32):
    """Normalize characters for classification."""
    raise Exception("disabled")
    if amax(image)<1e-3: return zeros((size,size))
    cimage = array(image*1.0/amax(image),'f')
    cimage = isotropic_rescale(cimage,size)
    cimage = csnormalize(cimage)
    return cimage

@checks(PATCH,size=int,scale=float,bar=object)
def line_normalize(image,size=32,scale=1.0,bar=None):
    """Normalize a character based on line geometry."""
    centroid = measurements.center_of_mass(image)
    return extract_centered_scaled_barred(image,shaped(size),centroid,scale,bar=bar)

