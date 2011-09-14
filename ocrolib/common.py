import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology

import iulib
import utils
from utils import allsplitext,write_text
import docproc
import ligatures
import fstutils
import openfst
import ocrofst
import ocrorast
import ocrolseg
import ocropreproc

import cPickle as pickle
pickle_mode = 2

################################################################
### deprecated
################################################################

import ocropus
import ocroold

################################################################
### exceptions
################################################################

class Unimplemented():
    def __init__(self,s):
        Exception.__init__(self,inspect.stack()[1][3])

class BadClassLabel(Exception):
    def __init__(self,s):
        Exception.__init__(self,s)

def check_valid_class_label(s):
    if type(s)==unicode:
        if re.search(r'[\0-\x20]',s):
            raise BadClassLabel(s)
    elif type(s)==str:
        if re.search(r'[^\x21-\x7e]',s):
            raise BadClassLabel(s)
    else:
        raise BadClassLabel(s)

def summary(x):
    if type(x)==numpy.ndarray:
        return "<ndarray %s %s>"%(x.shape,x.dtype)
    if type(x)==str and len(x)>10:
        return '"%s..."'%x
    if type(x)==list and len(x)>10:
        return '%s...'%x
    return str(x)

class RecognitionError(Exception):
    def __init__(self,explanation,**kw):
        self.context = kw
        s = [explanation]
        s += ["%s=%s"%(k,summary(kw[k])) for k in kw]
        message = " ".join(s)
        Exception.__init__(self,message)

################################################################
### file name manipulation
################################################################

def expand_args(args):
    if len(args)==1 and os.path.isdir(args[0]):
        return sorted(glob.glob(args[0]+"/????/??????.png"))
    else:
        return args

class OcropusFileNotFound:
    def __init__(self,fname):
        self.fname = fname
    def __str__(self):
        return "<OcropusFileNotFound "+self.fname+">"

data_paths = [
    ".",
    "./models",
    "./data",
    "./gui",
    "/usr/local/share/ocropus/models",
    "/usr/local/share/ocropus/data",
    "/usr/local/share/ocropus/gui",
    "/usr/local/share/ocropus",
]

def ocropus_find_file(fname):
    """Search for OCRopus-related files in common OCRopus install
    directories (as well as the current directory)."""
    if os.path.exists(fname):
        return fname
    for path in data_paths:
        full = path+"/"+fname
        if os.path.exists(full): return full
    raise OcropusFileNotFound(fname)

def fexists(fname):
    if os.path.exists(fname): return fname
    return None

def gtext(fname):
    """Given a file name, determines the ground truth suffix."""
    g = re.search(r'\.(rseg|cseg)\.([^./]+)\.png$',fname)
    if g:
        return g.group(2)
    g = re.search(r'\.([^./]+)\.(rseg|cseg)\.png$',fname)
    if g:
        return g.group(1)
    g = re.search(r'\.([^./]+)\.(png|costs|fst|txt)$',fname)
    if g:
        return g.group(1)
    return ""

def fvariant(fname,kind,gt=None):
    """Find the file variant corresponding to the given file name.
    Possible fil variants are line (or png), rseg, cseg, fst, costs, and txt.
    Ground truth files have an extra suffix (usually something like "gt",
    as in 010001.gt.txt or 010001.rseg.gt.png).  By default, the variant
    with the same ground truth suffix is produced.  The non-ground-truth
    version can be produced with gt="", the ground truth version can
    be produced with gt="gt" (or some other desired suffix)."""
    if gt is None:
        gt = gtext(fname)
    elif gt!="":
        gt = "."+gt
    base,ext = allsplitext(fname)
    if kind=="line" or kind=="png":
        return base+gt+".png"
    if kind=="rseg":
        return base+".rseg"+gt+".png"
    if kind=="cseg":
        return base+".cseg"+gt+".png"
    if kind=="costs":
        return base+gt+".costs"
    if kind=="fst":
        return base+gt+".fst"
    if kind=="txt":
        return base+gt+".txt"
    raise Exception("unknown kind: %s"%kind)

def fcleanup(fname,gt,kinds):
    for kind in kinds:
        s = fvariant(fname,kind,gt)
        if os.path.exists(s): os.unlink(s)

def ffind(fname,kind,gt=None):
    """Like fvariant, but throws an IOError if the file variant
    doesn't exist."""
    s = fvariant(fname,kind,gt=gt)
    if not os.path.exists(s):
        raise IOError(s)
    return s

def fopen(fname,kind,gt=None,mode="r"):
    """Like fvariant, but opens the file."""
    return open(fvariant(fname,kind,gt),mode)


def set_params(object,kw,warn=1):
    """Given an object and a dictionary of keyword arguments,
    set only those object properties that are already instance
    variables of the given object.  Returns a new dictionary
    without the key,value pairs that have been used.  If
    all keywords have been used, afterwards, len(kw)==0."""
    kw = kw.copy()
    for k,v in kw.items():
        if hasattr(object,k):
            setattr(object,k,v)
            del kw[k]
    return kw

def caller():
    frame = sys._getframe(2)
    info = inspect.getframeinfo(frame)
    result = "%s:%d (%s)"%(info.filename,info.lineno,info.function)
    del frame
    return result

def logging(message,*args):
    """Write a log message (to stderr by default)."""
    message = message%args
    sys.stderr.write(message)

def die(message,*args):
    """Die with an error message."""
    message = message%args
    message = caller()+" FATAL "+message+"\n"
    sys.stderr.write(message)
    sys.exit(1)

def warn(message,*args):
    """Give a warning message."""
    message = message%args
    message = caller()+" WARNING "+message+"\n"
    sys.stderr.write(message)

already_warned = {}

def warn_once(message,*args):
    """Give a warning message, but just once."""
    c = caller()
    if c in already_warned: return
    already_warned[c] = 1
    message = message%args
    message = c+" WARNING "+message+"\n"
    sys.stderr.write(message)

def quick_check_page_components(page_bin,dpi):
    """Quickly check whether the components of page_bin are
    reasonable.  Returns a value between 0 and 1; <0.5 means that
    there is probably something wrong."""
    return 1.0

def quick_check_line_components(line_bin,dpi):
    """Quickly check whether the components of line_bin are
    reasonable.  Returns a value between 0 and 1; <0.5 means that
    there is probably something wrong."""
    return 1.0

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning,stacklevel=2)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc

################################################################
### conversion functions
################################################################

def ustrg2unicode(u,lig=ligatures.lig):
    """Convert an iulib ustrg to a Python unicode string; the
    C++ version iulib.ustrg2unicode does weird things for special
    symbols like -3"""
    result = ""
    for i in range(u.length()):
        value = u.at(i)
        if value>=0:
            c = lig.chr(value)
            if c is not None:
                result += c
            else:
                result += "<%d>"%value
    return result

def isfp(a):
    """Check whether the array is a floating point array."""
    if type(a)==str:
        if a in ['f','d']: return 1
        else: return 0
    if type(a)==iulib.floatarray: return 1
    try:
        if a.dtype in [dtype('f'),dtype('d')]: return 1
    except:
        pass
    return 0

def checknp(a):
    """Checks whether the argument is a numpy array.  Raises an error if not."""
    if type(a) in [iulib.bytearray,iulib.intarray,iulib.floatarray,iulib.rectarray]:
        raise Exception("numpy array expected; an narray was passed")
    assert type(a)==numpy.ndarray
def checkna(a):
    """Checks whether the argument is an narray.  Raises an error if not."""
    if type(a) in [iulib.bytearray,iulib.intarray,iulib.floatarray,iulib.rectarray]:
        return
    if type(a)==numpy.array:
        raise Exception("narray expected; a numpy array was passed")
    raise Exception("expected an narray, got something different")

def ctype(a):
    """Return the numpy type character for an array."""
    if type(a)==str: return a
    if type(a)==iulib.floatarray: return 'f'
    if type(a)==iulib.intarray: return 'i'
    if type(a)==iulib.bytearray: return 'B'
    return a.dtype

def numpy2narray(page,type='B'):
    """Convert a numpy image to an narray. Flips from raster to
    mathematical coordinates.  When converting float to integer
    types, multiplies with 255.0, and when converting integer to
    float types, divides by 255.0."""
    checknp(page)
    if type is None: type = ctype(page)
    if isfp(page) and not isfp(type):
        page = array(255*page,dtype='B')
    elif not isfp(page) and isfp(type):
        page = page/255.0
    page = page.transpose([1,0]+range(2,page.ndim))[:,::-1,...]
    return iulib.narray(page,type=type)

def narray2numpy(na,type='B'):
    """Convert an narray image to a numpy image. Flips from mathematical
    coordinates to raster coordinates.  When converting integer to float
    types, multiplies with 255.0, and when converting integer to float
    types divides by 255.0"""
    checkna(na)
    if type is None: type = ctype(na)
    if isfp(na) and not isfp(type):
        page = iulib.numpy(na,'f')
        page = array(255.0*page,dtype=type)
    elif not isfp(na) and isfp(type):
        page = iulib.numpy(na,type=type)
        page /= 255.0
    else:
        page = iulib.numpy(na,type=type)
    return page.transpose([1,0]+range(2,page.ndim))[::-1,...]

def vector2narray(v,type='f'):
    """Convert a numpy vector to an narray.  If ndim>1, it converts to
    mathematical coordinates.  This is used with classifiers."""
    checknp(v)
    if v.ndim==1: return iulib.narray(v,type='f')
    else: return iulib.narray(v[::-1,...].transpose([1,0]+range(2,v.ndim)))

def narray2vector(na,type='f'):
    """Convert an narray vector to numpy.  If ndim>1, it converts to
    raster coordinates.  This is used with classifiers."""
    a = iulib.numpy(na)
    if a.ndim>1: return a.transpose([1,0]+range(2,a.ndim))[::-1,...]
    else: return a

def page2narray(page,type='B'):
    """Convert page images to narrays."""
    checknp(page)
    return numpy2narray(page,type=type)

def narray2page(page,type='B'):
    """Convert narrays to page images."""
    checkna(page)
    return narray2numpy(page,type=type)

def line2narray(line,type='B'):
    """Convert line images to narrays."""
    checknp(line)
    return numpy2narray(line,type=type)

def narray2line(line,type='B'):
    """Convert line narrays to line images."""
    checkna(line)
    return narray2numpy(line,type=type)

def narray2pseg(na):
    """Convert an narray to a page segmentation (rank 3, RGB)."""
    checkna(na)
    pseg = iulib.numpy(na,type='i')
    pseg = array([pseg>>16,pseg>>8,pseg],'B')
    pseg = transpose(pseg,[2,1,0])
    pseg = pseg[::-1,...]
    return pseg

def pseg2narray(pseg):
    """Convert a page segmentation (rank 3, RGB) to an narray."""
    checknp(pseg)
    assert pseg.dtype=='B' and pseg.ndim==3
    r = numpy2narray(ascontiguousarray(pseg[:,:,0]))
    g = numpy2narray(ascontiguousarray(pseg[:,:,1]))
    b = numpy2narray(ascontiguousarray(pseg[:,:,2]))
    rgb = iulib.intarray()
    iulib.pack_rgb(rgb,r,g,b)
    return rgb

def narray2lseg(na):
    """Convert an narray to a line segmentation."""
    checkna(na)
    pseg = iulib.numpy(na,type='i')
    pseg = transpose(pseg,[1,0])
    pseg = pseg[::-1,...]
    return pseg

def lseg2narray(lseg):
    """Convert a line segmentation (rank 2, 'i') to an narray."""
    checknp(lseg)
    assert lseg.dtype=='i' and lseg.ndim==2,"wanted rank 2 'i' array, got %s"%lseg
    lseg = lseg[::-1,...].transpose()
    lseg = iulib.narray(lseg,type='i')
    return lseg

def rect2raster(r,h):
    """Convert iulib rectangles to raster coordinates.  Raster coordinates are given
    as (row0,col0,row1,col1).  Note that this is different from some other parts of
    Python, which transpose the rows and columns."""
    (x0,y0,x1,y1) = (r.x0,r.y0,r.x1,r.y1)
    y1 = h-y1-1
    y0 = h-y0-1
    return (y1,x0,y0,x1)

def raster2rect(r,h):
    """Convert raster coordinates (row,col,row,col) to iulib
    rectangles.  Input is (row0,col0,row1,col1)."""
    (r0,c0,r1,c1) = r
    return iulib.rectangle(c0,h-r1-1,c1,h-r0-1)

def rect2math(r):
    """Convert rectangles to mathematical coordinates."""
    return (r.x0,r.y0,r.x1,r.y1)

def math2rect(r):
    """Convert mathematical coordinates to rectangle coordinates."""
    (x0,y0,x1,y1) = r
    return iulib.rectangle(x0,y0,x1,y1)

################################################################
### simple shape comparisons
################################################################

def thin(image,c=1):
    if c>0: image = morphology.binary_closing(image,iterations=c)
    image = array(image,'B')
    image = numpy2narray(image)
    iulib.thin(image)
    return array(narray2numpy(image),'B')

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

def cut(image,box,margin=0,bg=0,dtype=None):
    (r0,c0,r1,c1) = box
    r0 -= margin; c0 -= margin; r1 += margin; c1 += margin
    if dtype is None: dtype = image.dtype
    result = interpolation.shift(image,(-r0,-c0),output=dtype,order=0,cval=bg)
    return result[:(r1-r0),:(c1-c0)]

class RegionExtractor:
    """A class facilitating iterating over the parts of a segmentation."""
    def __init__(self):
        self.comp = ocropus.RegionExtractor()
        self.cache = {}
    def clear(self):
        del self.cache
        self.cache = {}
    def setImage(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'."""
        self.h = image.shape[0]
        self.comp.setImage(self.pseg2narray(image))
    def setImageMasked(self,image,mask,lo,hi):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This picks a subset of the segmentation to iterate
        over, using a mask and lo and hi values.."""
        self.h = image.shape[0]
        assert type(mask)==int and type(lo)==int and type(hi)==int
        self.comp.setImage(self.pseg2narray(image),mask,lo,hi)
    def setPageColumns(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This iterates over the columns."""
        self.h = image.shape[0]
        image = pseg2narray(image)
        self.comp.setPageColumns(self,image)
    def setPageParagraphs(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This iterates over the paragraphs (if present
        in the segmentation)."""
        self.h = image.shape[0]
        image = pseg2narray(image)
        self.comp.setPageParagraphs(self,image)
    def setPageLines(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This iterates over the lines."""
        self.h = image.shape[0]
        image = pseg2narray(image)
        # iulib.write_image_packed("_seg.png",image)
        self.comp.setPageLines(image)
    def id(self,i):
        """Return the RGB pixel value for this segment."""
        return self.comp.id(i)
    def x0(self,i):
        """Return x0 (column) for the start of the box."""
        return self.comp.x0(i)
    def x1(self,i):
        """Return x0 (column) for the end of the box."""
        return self.comp.x1(i)
    def y0(self,i):
        """Return y0 (row) for the start of the box."""
        return h-self.comp.y1(i)-1
    def y1(self,i):
        """Return y0 (row) for the end of the box."""
        return h-self.comp.y0(i)-1
    def bbox(self,i):
        """Return the bounding box in raster coordinates
        (row0,col0,row1,col1)."""
        r = self.comp.bbox(i)
        return rect2raster(r,self.h)
    def bboxMath(self,i):
        """Return the bounding box in math coordinates
        (row0,col0,row1,col1)."""
        r = self.comp.bbox(i)
        return rect2math(r)
    def length(self):
        """Return the number of components."""
        return self.comp.length()
    def mask(self,index,margin=0):
        """Return the mask for component index."""
        result = iulib.bytearray()
        self.comp.mask(result,index,margin)
        return narray2numpy(result)
    def extract(self,image,index,margin=0):
        """Return the subimage for component index."""
        h,w = image.shape[:2]
        (r0,c0,r1,c1) = self.bbox(index)
        mask = self.mask(index,margin=margin)
        return image[max(0,r0-margin):min(h,r1+margin),max(0,c0-margin):min(w,c1+margin),...]
    def extractMasked(self,image,index,grow,bg=None,margin=0,dtype=None):
        """Return the masked subimage for component index, elsewhere the bg value."""
        if bg is None: bg = amax(image)
        h,w = image.shape[:2]
        mask = self.mask(index,margin=margin)
        mh,mw = mask.shape
        box = self.bbox(index)
        r0,c0,r1,c1 = box
        subimage = cut(image,(r0,c0,r0+mh-2*margin,c0+mw-2*margin),margin,bg=bg)
        return where(mask,subimage,bg)

class Grouper(ocroold.CommonComponent):
    """Perform grouping operations on segmented text lines, and
    create a finite state transducer for classification results."""
    c_interface = "IGrouper"
    def setSegmentation(self,segmentation):
        """Set the line segmentation."""
        self.comp.setSegmentation(lseg2narray(segmentation))
        self.h = segmentation.shape[0]
    def setCSegmentation(self,segmentation):
        """Set the line segmentation, assumed to be a cseg."""
        self.comp.setCSegmentation(lseg2narray(segmentation))
        self.h = segmentation.shape[0]
    def length(self):
        """Number of groups."""
        return self.comp.length()
    def getMask(self,i,margin=0):
        """Get the mask image for group i."""
        if self.isEmpty(i): return None
        rect = rectangle()
        mask = iulib.bytearray()
        self.comp.getMask(rect,mask,i,margin)
        return (rect2raster(rect,self.h),narray2numpy(mask,'f'))
    def getMaskAt(self,i,rect):
        """Get the mask for group i and contained in the given rectangle."""
        if self.isEmpty(i): return None
        rect = raster2rect(rect,self.h)
        mask = iulib.bytearray()
        self.comp.getMaskAt(mask,i,rect)
        return narray2numpy(mask,'f')
    def isEmpty(self,i):
        y0,x0,y1,x1 = self.boundingBox(i)
        return y0>=y1 or x0>=x1
    def boundingBox(self,i):
        """Get the bounding box for group i."""
        return rect2raster(self.comp.boundingBox(i),self.h)
    def bboxMath(self,i):
        """Get the bounding box for group i."""
        return rect2math(self.comp.boundingBox(i))
    def start(self,i):
        """Get the identifier of the character segment starting this group."""
        return self.comp.start(i)
    def end(self,i):
        """Get the identifier of the character segment ending this group."""
        return self.comp.end(i)
    def getSegments(self,i):
        """Get a list of all the segments making up this group."""
        l = iulib.intarray()
        self.comp.getSegments(l,i)
        return [l.at(i) for i in range(l.length())]
    def extract(self,source,dflt,i,grow=0,dtype='f'):
        """Extract the image corresponding to group i.  Background pixels are
        filled in with dflt."""
        if self.isEmpty(i): return None
        checknp(source)
        if isfp(source):
            out = iulib.floatarray()
            self.comp.extract(out,numpy2narray(source,'f'),dflt,i,grow)
            return narray2numpy(out,'f')
        else:
            out = iulib.bytearray()
            self.comp.extract(out,numpy2narray(source,'B'),dflt,i,grow)
            return narray2numpy(out,'B')
    def extractWithMask(self,source,i,grow=0):
        """Extract the image and mask corresponding to group i"""
        if self.isEmpty(i): return None
        checknp(source)
        if isfp(source):
            out = iulib.floatarray()
            mask = iulib.bytearray()
            self.comp.extractWithMask(out,mask,numpy2narray(source,'f'),i,grow)
            return (narray2numpy(out,'f'),narray2numpy(out,'b'))
        else:
            out = iulib.bytearray()
            mask = iulib.bytearray()
            self.comp.extractWithMask(out,mask,numpy2narray(source,'B'),i,grow)
            return (narray2numpy(out,'B'),narray2numpy(out,'B'))
    def extractSliced(self,source,dflt,i,grow=0):
        """Extract the image and mask corresponding to group i, slicing through the entire input
        line.  Background pixels are filled with dflt."""
        if self.isEmpty(i): return None
        if isfp(source):
            out = iulib.floatarray()
            self.comp.extractSliced(out,numpy2narray(source,'f'),dflt,i,grow)
            return narray2numpy(out,'f')
        else:
            out = iulib.bytearray()
            self.comp.extractSliced(out,numpy2narray(source,'B'),dflt,i,grow)
            return narray2numpy(out,'B')
    def extractSlicedWithMask(self,source,i,grow=0):
        """Extract the image and mask corresponding to group i, slicing through the entire
        input line."""
        if self.isEmpty(i): return None
        if isfp(source):
            out = iulib.floatarray()
            mask = iulib.bytearray()
            self.comp.extractWithMask(out,mask,numpy2narray(source,'f'),i,grow)
            return (narray2numpy(out,'f'),narray2numpy(out,'B'))
        else:
            out = iulib.bytearray()
            mask = iulib.bytearray()
            self.comp.extractWithMask(out,mask,numpy2narray(source,'B'),i,grow)
            return (narray2numpy(out,'B'),narray2numpy(out,'B'))
    def setClass(self,i,cls,cost):
        """Set the class for group i, and the associated cost.  The class may
        be given as an integer, as a string, or as a unicode string.  The cost
        should be non-negative."""
        cost = float(cost)
        if type(cls)==str:
            u = iulib.unicode2ustrg(unicode(cls))
            self.comp.setClass(i,u,cost)
        elif type(cls)==unicode:
            u = iulib.unicode2ustrg(cls)
            self.comp.setClass(i,u,cost)
        elif type(cls)==int:
            assert cls>=-3,"bad cls: %d (should be >= -3)"%cls
            self.comp.setClass(i,cls,cost)
        else:
            raise Exception("bad class type '%s'"%cls)
    def setSpaceCost(self,i,yes_cost,no_cost):
        """Set the cost of putting a space or not putting a space after
        group i."""
        self.comp.setSpaceCost(i,yes_cost,no_cost)
    def getLattice(self):
        """Construct the lattice for the group, using the setClass and setSpaceCost information."""
        fst = ocrofst.OcroFST()
        self.comp.getLattice(fst.comp)
        return fst
    def clearLattice(self):
        """Clear all the lattice-related information accumulated so far."""
        self.comp.clearLattice()
    def pixelSpace(self,i):
        """???"""
        return self.comp.pixelSpace(i)
    def setSegmentationAndGt(self,rseg,cseg,gt):
        """Set the line segmentation."""
        assert rseg.shape==cseg.shape
        self.gt = gt
        u = iulib.ustrg()
        u.assign("?"*len(gt))
        self.comp.setSegmentationAndGt(lseg2narray(rseg),lseg2narray(cseg),u)
        self.h = rseg.shape[0]
    def getGtIndex(self,i):
        return self.comp.getGtIndex(i)
    def getGtClass(self,i):
        index = self.getGtIndex(i)
        if index<0: return "~"
        return self.gt[index-1]

class StandardGrouper(Grouper):
    """The grouper usually used for printed OCR."""
    c_class = "StandardGrouper"

### native code image I/O; we use this because it works more reliably
### than Python's code for TIFF files

### FIXME eventually try to replace this with builtin code

def iulib_page_iterator(files):
    """Given a list of files, iterate through the page images in those
    files.  When multi-page TIFF files are encountered, iterates through
    each such TIFF file.  Yields tuples (image,filename), where image
    is in iulib format."""
    for file in files:
        _,ext = os.path.splitext(file)
        if ext.lower()==".tif" or ext.lower()==".tiff":
            if os.path.getsize(file)>2e9:
                raise IOError("TIFF file is greater than 2G")
            tiff = iulib.Tiff(file,"r")
            for i in range(tiff.numPages()):
                image = iulib.bytearray()
                try:
                    tiff.getPageRaw(image,i,True)
                except:
                    tiff.getPage(image,i,True)
                yield image,"%s[%d]"%(file,i)
        else:
            image = iulib.bytearray()
            iulib.read_image_gray(image,file)
            yield image,file

def page_iterator(files):
    """Given a list of files, iterate through the page images in those
    files.  When multi-page TIFF files are encountered, iterates through
    each such TIFF file.  Yields tuples (image,filename), where image
    is in NumPy format."""
    # use the iulib implementation because its TIFF reader actually works;
    # the TIFF reader in all the Python libraries is broken
    for image,file in iulib_page_iterator(files):
        yield narray2page(image),file

def read_image_gray(file,type='B'):
    """Read an image in grayscale."""
    if not os.path.exists(file): raise IOError(file)
    na = iulib.bytearray()
    iulib.read_image_gray(na,file)
    return narray2numpy(na,type=type)

def write_image_gray(file,image):
    """Write an image in grayscale."""
    assert (array(image.shape)>0).all()
    if image.ndim==2:
        iulib.write_image_gray(file,numpy2narray(image))
    elif image.ndim==3:
        iulib.write_image_gray(file,numpy2narray(mean(image,axis=2)))
    else:
        raise Exception("ndim must be 2 or 3, not %d"%image.ndim)

def draw_pseg(pseg,axis=None):
    """Display a pseg."""
    if axis is None:
        axis = subplot(111)
    h = pseg.dim(1)
    regions = RegionExtractor()
    regions.setPageLines(pseg)
    for i in range(1,regions.length()):
        (r0,c0,r1,c1) = (regions.x0(i),regions.y0(i),regions.x1(i),regions.y1(i))
        p = patches.Rectangle((c0,r0),c1-c0,r1-r0,edgecolor="red",fill=0)
        axis.add_patch(p)

def write_page_segmentation(name,pseg,white=1):
    """Write a numpy page segmentation (rank 3, type='B' RGB image.)"""
    pseg = pseg2narray(pseg)
    if white: ocropus.make_page_segmentation_white(pseg)
    iulib.write_image_packed(name,pseg)
    
def read_page_segmentation(name,black=1):
    """Write a numpy page segmentation (rank 3, type='B' RGB image.)"""
    if not os.path.exists(name): raise IOError(name)
    pseg = iulib.intarray()
    iulib.read_image_packed(pseg,name)
    if black: ocropus.make_page_segmentation_black(pseg)
    return narray2pseg(pseg)
    
def write_line_segmentation(name,lseg,white=1):
    """Write a numpy line segmentation."""
    lseg = lseg2narray(lseg)
    if white: ocropus.make_line_segmentation_white(lseg)
    iulib.write_image_packed(name,lseg)
    
def read_line_segmentation(name,black=1):
    """Write a numpy line segmentation."""
    if not os.path.exists(name): raise IOError(name)
    lseg = iulib.intarray()
    iulib.read_image_packed(lseg,name)
    if black: ocropus.make_line_segmentation_black(lseg)
    return narray2lseg(lseg)

def renumber_labels(line,start):
    line = lseg2narray(line)
    iulib.renumber_labels(line,start)
    return narray2lseg(line)

### code for instantiation native components

def pyconstruct(s):
    """Constructs a Python object from a constructor, an expression
    of the form x.y.z.name(args).  This ensures that x.y.z is imported.
    In the future, more forms of syntax may be accepted."""
    env = {}
    if "(" not in s:
        s += "()"
    path = s[:s.find("(")]
    if "." in path:
        module = path[:path.rfind(".")]
        print "import",module
        exec "import "+module in env
    return eval(s,env)

def mkpython(name):
    """Tries to instantiate a Python class.  Gives an error if it looks
    like a Python class but can't be instantiated.  Returns None if it
    doesn't look like a Python class."""
    if type(name) is not str:
        return name()
    elif name[0]=="=":
        return pyconstruct(name[1:])
    elif "(" in name or "." in name:
        return pyconstruct(name)
    else:
        return None

def make_ICleanupGray(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or CleanupGray().make(name)
def make_ICleanupBinary(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or CleanupBinary().make(name)
def make_IBinarize(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or Binarize().make(name)
def make_ITextImageClassification(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or TextImageClassification().make(name)
def make_ISegmentPage(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or SegmentPage().make(name)
def make_ISegmentLine(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or SegmentLine().make(name)
def make_IGrouper(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or Grouper().make(name)
def make_IRecognizeLine(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or RecognizeLine().make(name)
def make_IModel(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or Model().make(name)
def make_IExtractor(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    return mkpython(name) or Extractor().make(name)

### native feature extraction code

def hole_counts(image,r=1.0):
    """Count the number of holes in the input image.  This assumes
    background-is-FIXME."""
    image = binarize_range(image)
    return ocropus.hole_counts(numpy2narray(image),r)

def component_counts(image,r=1.0):
    """Count the number of connected components in the image.  This
    assumes background-is-FIXME."""
    image = binarize_range(image)
    return ocropus.component_counts(numpy2narray(image),r)

def junction_counts(image,r=1.0):
    """Count the number of junctions in the image.  This
    assumes background-is-FIXME."""
    image = binarize_range(image)
    return ocropus.junction_counts(numpy2narray(image),r)

def endpoints_counts(image,r=1.0):
    """Count the number of endpoints in the image.  This
    assumes background-is-FIXME."""
    image = binarize_range(image)
    return ocropus.endpoints_counts(numpy2narray(image),r)

################################################################
### alignment, segmentations, and conversions
################################################################

def intarray_as_unicode(a,skip0=1):
    result = u""
    for i in range(len(a)):
        if a[i]!=0:
            assert a[i]>=0 and a[i]<0x110000,"%d (0x%x) character out of range"%(a[i],a[i])
            result += unichr(a[i])
    return result
    
def read_lmodel_or_textlines(file):
    """Either reads a language model in .fst format, or reads a text file
    and corresponds a language model out of its lines."""
    if not os.path.exists(file): raise IOError(file)
    if file[-4:]==".fst":
        return ocrofst.OcroFST().load(file)
    else:
        import fstutils
        result = fstutils.load_text_file_as_fst(file)
        assert isinstance(result,ocrofst.OcroFST)
        return result

def OLD_rseg_map(inputs):
    """This takes an array of the input labels produced by a beam search.
    The input labels contain the correspondence
    between the rseg label and the character.  These are put into
    a dictionary and returned.  This is used for alignment between
    a segmentation and text."""
    n = len(inputs)
    segs = []
    for i in range(n):
        start = inputs[i]>>16
        end = inputs[i]&0xffff
        segs.append((start,end))
    n = amax([s[1] for s in segs])+1
    count = 0
    map = zeros(n,'i')
    for i in range(len(segs)):
        start,end = segs[i]
        if start==0 or end==0: continue
        count += 1
        for j in range(start,end+1):
            map[j] = count
    return map

def OLD_recognize_and_align(image,linerec,lmodel,beam=1000,nocseg=0):
    """Perform line recognition with the given line recognizer and
    language model.  Outputs an object containing the result (as a
    Python string), the costs, the rseg, the cseg, the lattice and the
    total cost.  The recognition lattice needs to have rseg's segment
    numbers as inputs (pairs of 16 bit numbers); SimpleGrouper
    produces such lattices.  cseg==None means that the connected
    component renumbering failed for some reason."""

    assert isinstance(linerec,RecognizeLine)

    lattice,rseg = linerec.recognizeLineSeg(image)
    v1,v2,ins,outs,costs = beam_search(lattice,lmodel,beam)
    result = intarray_as_unicode(outs,skip0=0)

    # compute the cseg
    rmap = rseg_map(ins)
    n = len(rseg)
    if not nocseg and len(rmap)>1:
        r,c = rseg.shape
        cseg = zeros((r,c),'i')
        for i in range(r):
            for j in range(c):
                value = rseg[i,j]
                cseg[i,j] = rmap[value]
    else:
        cseg = None

    # return everything we computed
    return utils.Record(image=image,
                        output=result,
                        raw=outs,
                        costs=costs,
                        rseg=rseg,
                        cseg=cseg,
                        lattice=lattice,
                        cost=sum(costs))

def rect_union(rectangles):
    if len(rectangles)<1: return (0,0,-1,-1)
    r = array(rectangles)
    return (amin(r[:,0]),amax(r[:,0]),amin(r[:,1]),amax(r[:1]))

def compute_alignment(lattice,rseg,lmodel,beam=1000,verbose=0,lig=ligatures.lig):
    """Given a lattice produced by a recognizer, a raw segmentation,
    and a language model, computes the best solution, the cseg, and
    the corresponding costs.  These are returned as Python data structures.
    The recognition lattice needs to have rseg's segment numbers as inputs
    (pairs of 16 bit numbers); SimpleGrouper produces such lattices."""

    v1,v2,ins,outs,costs = beam_search(lattice,lmodel,beam)

    # useful for debugging

    if 0:
        for i in range(len(ins)):
            print i,ins[i]>>16,ins[i]&0xffff,lig.chr(outs[i]),costs[i]

    assert len(ins)==len(outs)
    n = len(ins)

    # This is a little tricky because we need to deal with ligatures.
    # For any transition followed by epsilon transitions on the
    # output, we group all the segments of the epsilon transition with
    # the preceding non-epsilon transition.

    result_l = [""]
    costs_l = [0.0]
    segs = [(-1,-1)]

    i = 0
    while i<n:
        j = i+1
        start = ins[i]>>16
        end = ins[i]&0xffff
        cls = [outs[i]]
        # print "  %4d (%2d,%2d) %3d %s"%(i,start,end,outs[i],unichr(outs[i]))
        # while j<n and ((ins[j]==0 and outs[j]!=32) or outs[j]==0):
        while j<n and outs[j]==0:
            # print " +%4d (%2d,%2d) %3d %s"%(i,ins[j]>>16,ins[j]&0xffff,outs[j],unichr(outs[j]))
            if ins[j]!=0:
                start = min(start,ins[j]>>16)
                end = max(end,ins[j]&0xffff)
            if outs[j]!=0:
                cls.append(outs[j])
            j = j+1
        cls = "".join([lig.chr(x) for x in cls])
        if cls!="":
            result_l.append(cls)
            costs_l.append(sum(costs[i:j]))
            segs.append((start,end))
        i = j

    rseg_boxes = docproc.seg_boxes(rseg)

    # Now run through the segments and create a table that maps rseg
    # labels to the corresponding output element.

    assert len(result_l)==len(segs)
    assert len(costs_l)==len(segs)
    bboxes = []

    rmap = zeros(amax(rseg)+1,'i')
    for i in range(1,len(segs)):
        start,end = segs[i]
        if verbose: print i+1,start,end,"'%s'"%result[i],costs.at(i)
        if start==0 or end==0: continue
        rmap[start:end+1] = i
        bboxes.append(rect_union(rseg_boxes[start:end+1]))
    assert rmap[0]==0

    # Finally, to get the cseg, apply the rmap table from above.

    cseg = zeros(rseg.shape,'i')
    for i in range(cseg.shape[0]):
        for j in range(cseg.shape[1]):
            cseg[i,j] = rmap[rseg[i,j]]

    if 0:
        print len(rmap),rmap
        print len(segs),segs
        print len(result_l),result_l
        print len(costs_l),costs_l
        print amin(cseg),amax(cseg)

    # assert len(segs)==len(rmap) 
    assert len(segs)==len(result_l) 
    assert len(segs)==len(costs_l)
    return utils.Record(
        # alignment output; these all have the same lengths
        output_l=result_l,
        segs=segs,
        costs=array(costs_l,'f'),
        # other convenient output representation
        output="".join(result_l),
        output_t=fstutils.implode_transcription(result_l),
        cost=sum(costs_l),
        # raw beam search output
        ins=ins,
        outs=outs,
        # segmentation images
        rseg=rseg,
        cseg=cseg,
        # the lattice
        lattice=lattice,
        # bounding boxes
        bboxes=bboxes,
        )

def recognize_and_align(image,linerec,lmodel,beam=1000,nocseg=0,lig=ligatures.lig):
    """Perform line recognition with the given line recognizer and
    language model.  Outputs an object containing the result (as a
    Python string), the costs, the rseg, the cseg, the lattice and the
    total cost.  The recognition lattice needs to have rseg's segment
    numbers as inputs (pairs of 16 bit numbers); SimpleGrouper
    produces such lattices.  cseg==None means that the connected
    component renumbering failed for some reason."""

    lattice,rseg = linerec.recognizeLineSeg(image)
    v1,v2,ins,outs,costs = beam_search(lattice,lmodel,beam)
    result = compute_alignment(lattice,rseg,lmodel,beam=beam,lig=lig)
    return result

################################################################
### new, pure Python components
################################################################

class PyComponent:
    """Defines common methods similar to CommonComponent, but for Python
    classes. Use of this base class is optional."""
    def init(self):
        pass
    def name(self):
        return "%s"%self
    def description(self):
        return "%s"%self
    def set(self,**kw):
        kw = set_params(self,kw)
        assert kw=={},"extra params to %s: %s"%(self,kw)
    def pset(self,key,value):
        if hasattr(self,key):
            self.__dict__[key] = value
    def pget(self,key):
        return getattr(self,key)
    def pgetf(self,key):
        return float(getattr(self,key))
    
class ClassifierModel(PyComponent):
    """Wraps all the necessary functionality around a classifier in order to
    turn it into a character recognition model."""
    def __init__(self,**kw):
        self.nbest = 5
        self.minp = 1e-3
        self.classifier = self.makeClassifier()
        self.extractor = self.makeExtractor()
        kw = set_params(self,kw)
        kw = set_params(self.classifier,kw)
        self.rows = None
        self.nrows = 0
        self.classes = []
        self.c2i = {}
        self.i2c = []
        self.geo = None

    def set(self,**kw):
        kw = set_params(self,kw)
        kw = set_params(self.classifier,kw)
        assert kw=={},"extra parameters to %s: %s"%(self,kw)

    ## helper methods

    def setupGeometry(self,geometry):
        """Set the self.geo instance variable to remember whether this
        Model uses geometry or not."""
        if self.geo is None:
            if geometry is None: 
                self.geo = 0
            else:
                self.geo = len(array(geometry,'f'))
    def makeInput(self,image,geometry):
        """Given an image and geometry, compute and return a feature vector.
        This calls the feature extractor via self.extract(image)."""
        v = self.extractor.extract(image).ravel()
        if self.geo>0:
            if geometry is not None:
                geometry = array(geometry,'f')
                assert len(geometry)==self.geo
                v = concatenate([v,geometry])
        return v
    def makeOutputs(self,w):
        """Given an output vector w from a classifier, translate use the
        i2c array to translate this into a list [(class,probability),...]
        representing the string output of the character recognizer."""
        result = []
        indexes = argsort(-w)
        for i in indexes[:self.nbest]:
            if w[i]<self.minp: break
            result.append((self.i2c[i],w[i]))
        return result

    ## public methods
    
    def clear(self):
        """Completely clear the classifier"""
        self.rows = None
        self.classes = None
        self.nrows = 0

    def cadd(self,image,c,geometry=None):
        """Add a character to the model for training.  The image may be of variable size.
        c should be the corresponding class, a string.  If geometry is given, it must be
        given always and consistently."""
        check_valid_class_label(c)
        if self.geo is None: 
            # first time around, remember whether this classifier uses geometry
            self.setupGeometry(geometry)
        v = self.makeInput(image,geometry)
        assert amin(v)>=-1.2 and amax(v)<=1.2
        if self.nrows==0:
            self.rows = zeros((1000,len(v)),'int8')
        elif self.nrows==len(self.rows):
            n,d = self.rows.shape
            self.rows.resize((1000+n,d))
        self.rows[self.nrows,:] = 100.0*v
        if c not in self.c2i:
            self.c2i[c] = len(self.i2c)
            self.i2c.append(c)
        self.classes.append(self.c2i[c])
        self.nrows += 1

    def updateModel(self,*args,**kw):
        """Perform actual training of the model."""
        n,d = self.rows.shape
        self.rows.resize(self.nrows,d)
        self.classifier.train(self.rows,array(self.classes,'i'),*args,**kw)
        self.clear()

    def updateModel1(self,*args,**kw):
        """Perform training of the model.  This actually is an iterator and
        returns from time to time during training to allow saving of intermediate
        models.  Use as in "for progress in model.updateModel1(): print progress" """
        if not hasattr(self.classifier,"train1"):
            warn_once("no train1 method; just doing training in one step")
            self.updateModel(*args,**kw)
            return
        n,d = self.rows.shape
        self.rows.resize(self.nrows,d)
        for progress in self.classifier.train1(self.rows,array(self.classes,'i'),*args,**kw):
            yield progress
        self.clear()

    def coutputs(self,image,geometry=None):
        """Given an image and corresponding geometry (as during
        training), compute outputs (posterior probabilities or
        discriminant functions) for the image.  Returns a list like
        [(class,probability),...]"""
        assert (not self.geo) or (geometry is not None),\
            "classifier requires geometry but none is given"
        v = self.makeInput(image,geometry)
        w = self.classifier.outputs(v.reshape(1,len(v)))[0]
        return self.makeOutputs(w)

    def cclassify(self,v,geometry=None):
        """Given an image and corresponding geometry (as during
        training), classify the image.  Returns just the class."""
        assert (not self.geo) or (geometry is not None),\
            "classifier requires geometry but none is given"
        v = self.makeInput(image,geometry)
        w = self.classifier.outputs(v.reshape(1,len(v)))[0]
        return self.i2c[argmax(w)]

    def coutputs_batch(self,images,geometries=None):
        """Given a list of images (and an optional list of geometries if the
        classifier requires it), compute a list of the corresponding outputs.
        This is the same as calling coutputs repeatedly, but it may be
        parallelized."""
        # FIXME parallelize this
        if geometries is None: geometries = [None]*len(images)
        result = []
        for i in range(len(images)):
            try:
                output = self.coutputs(images[i],geometries[i]) 
            except:
                print "recognition failed"
                output = []
            result.append(output)
        return result

    def save_component(self,path):
        """Save this component to a file (using pickle). Use
        ocrolib.load_component or cPickle.load to read the component
        back in again."""
        rows = self.rows
        nrows = self.nrows
        classes = self.classes
        self.rows = None
        self.classes = None
        self.nrows = None
        with open(path,"wb") as stream:
            pickle.dump(self,stream,pickle_mode)
        self.rows = rows
        self.classes = classes
        self.nrows = nrows

class BboxFE(PyComponent):
    """A feature extractor that only rescales the input image to fit into
    a 32x32 (or, generally, r x r box) and normalizes the vector.
    Parameters are r (size of the rescaled image), and normalize (can be
    one of "euclidean", "max", "sum", or None)."""
    def __init__(self,**kw):
        self.r = 32
        self.normalize = None
        set_params(self,kw)
    def extract(self,image):
        v = array(docproc.isotropic_rescale(image,self.r),'f')
        if not hasattr(self,"normalize") or self.normalize is None:
            pass
        elif self.normalize=="euclidean":
            v /= sqrt(sum(v**2))
        elif self.normalize=="max":
            v /= amax(v)
        elif self.normalize=="sum":
            v /= sum(abs(v))
        return v

class Classifier(PyComponent):
    """An abstraction for a classifier.  This gets trained on training vectors and
    returns vectors of posterior probabilities (or some other discriminant function.)
    You usually save these objects by pickling them."""
    def train(self,data,classes):
        """Train the classifier on the given dataset."""
        raise Unimplemented()
    def outputs(self,data):
        """Compute the ouputs corresponding to each input data vector."""
        raise Unimplemented()

################################################################
### loading and saving components
################################################################

# This code has to deal with a lot of special cases for all the
# different formats we have accrued.

def obinfo(ob):
    result = str(ob)
    if hasattr(ob,"shape"): 
        result += " "
        result += str(ob.shape)
    return result

def save_component(file,object,verbose=0,verify=0):
    """Save an object to disk in an appropriate format.  If the object
    is a wrapper for a native component (=inherits from
    CommonComponent and has a comp attribute, or is in package
    ocropus), write it using ocropus.save_component in native format.
    Otherwise, write it using Python's pickle.  We could use pickle
    for everything (since the native components pickle), but that
    would be slower and more confusing."""
    if hasattr(object,"save_component"):
        object.save_component(file)
        return
    # FIXME -- get rid of this eventually
    if isinstance(object,ocroold.CommonComponent) and hasattr(object,"comp"):
        ocropus.save_component(file,object.comp)
        return
    if type(object).__module__=="ocropus":
        ocropus.save_component(file,object)
        return
    if verbose: 
        print "[save_component]"
    if verbose:
        for k,v in object.__dict__.items():
            print ":",k,obinfo(v)
    with open(file,"wb") as stream:
        pickle.dump(object,stream,pickle_mode)
    if verify:
        if verbose: 
            print "[trying to read it again]"
        with open(file,"rb") as stream:
            test = pickle.load(stream)

def load_component(file):
    """Load a component from disk.  If file starts with "@", it is
    taken as a Python expression and evaluated, but this can be overridden
    by starting file with "=".  Otherwise, the contents of the file are
    examined.  If it looks like a native component, it is loaded as a line
    recognizers if it can be identified as such, otherwise it is loaded
    with load_Imodel as a model.  Anything else is loaded with Python's
    pickle.load."""

    if file[0]=="=":
        return pyconstruct(file[1:])
    elif file[0]=="@":
        file = file[1:]
    # FIXME -- get rid of this eventually
    with open(file,"r") as stream:
        start = stream.read(128)
    # FIXME -- get rid of this eventually
    if start.startswith("<object>\nlinerec\n"):
        result = RecognizeLine()
        result.comp = ocropus.load_IRecognizeLine(file)
        return result
    # FIXME -- get rid of this eventually
    if start.startswith("<object>"):
        result = Model()
        result.comp = ocropus.load_IModel(file)
        return result
    with open(file,"rb") as stream:
        return pickle.load(stream)

def load_linerec(file,wrapper=ocroold.CmodelLineRecognizer):
    component = load_component(file)
    if hasattr(component,"recognizeLine"):
        return component
    if hasattr(component,"coutputs"):
        return wrapper(cmodel=component)
    raise Exception("wanted linerec, got %s"%component)

def binarize_range(image,dtype='B'):
    threshold = (amax(image)+amin(image))/2
    scale = 1
    if dtype=='B': scale = 255
    return array(scale*(image>threshold),dtype=dtype)

def edit_distance(s,t,use_space=0,case_sensitive=0):
    if not case_sensitive:
        s = s.upper()
        t = t.upper()
    if not use_space:
        s = re.sub(r'\s+','',s)
        t = re.sub(r'\s+','',t)
    s_ = iulib.ustrg()
    s_.assign(s.encode("utf-8"))
    t_ = iulib.ustrg()
    t_.assign(t.encode("utf-8"))
    # print s_.as_string()
    # print t_.as_string()
    return ocropus.edit_distance(s_,t_)

