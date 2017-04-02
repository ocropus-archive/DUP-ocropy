# -*- coding: utf-8 -*-
################################################################
### common functions for data structures, file name manipulation, etc.
################################################################

from __future__ import print_function

import os
import os.path
import re
import sys
import sysconfig
import unicodedata
import inspect
import glob
import cPickle
from ocrolib.exceptions import (BadClassLabel, BadInput, FileNotFound,
                                OcropusException)

import numpy
from numpy import (amax, amin, array, bitwise_and, clip, dtype, mean, minimum,
                   nan, sin, sqrt, zeros)
from scipy.ndimage import morphology, measurements
import PIL

from default import getlocal
from toplevel import (checks, ABINARY2, AINT2, AINT3, BOOL, DARKSEG, GRAYSCALE,
                      LIGHTSEG, LINESEG, PAGESEG)
import chars
import codecs
import ligatures
import lstm
import morph
import multiprocessing
import sl

pickle_mode = 2


################################################################
# text normalization
################################################################

def normalize_text(s):
    """Apply standard Unicode normalizations for OCR.
    This eliminates common ambiguities and weird unicode
    characters."""
    s = unicode(s)
    s = unicodedata.normalize('NFC',s)
    s = re.sub(ur'\s+(?u)',' ',s)
    s = re.sub(ur'\n(?u)','',s)
    s = re.sub(ur'^\s+(?u)','',s)
    s = re.sub(ur'\s+$(?u)','',s)
    for m,r in chars.replacements:
        s = re.sub(unicode(m),unicode(r),s)
    return s

def project_text(s,kind="exact"):
    """Project text onto a smaller subset of characters
    for comparison."""
    s = normalize_text(s)
    s = re.sub(ur'( *[.] *){4,}',u'....',s) # dot rows
    s = re.sub(ur'[~_]',u'',s) # dot rows
    if kind=="exact":
        return s
    if kind=="nospace":
        return re.sub(ur'\s','',s)
    if kind=="spletdig":
        return re.sub(ur'[^A-Za-z0-9 ]','',s)
    if kind=="letdig":
        return re.sub(ur'[^A-Za-z0-9]','',s)
    if kind=="letters":
        return re.sub(ur'[^A-Za-z]','',s)
    if kind=="digits":
        return re.sub(ur'[^0-9]','',s)
    if kind=="lnc":
        s = s.upper()
        return re.sub(ur'[^A-Z]','',s)
    raise BadInput("unknown normalization: "+kind)

################################################################
### Text I/O
################################################################

def read_text(fname,nonl=1,normalize=1):
    """Read text. This assumes files are in unicode.
    By default, it removes newlines and normalizes the
    text for OCR processing with `normalize_text`"""
    with codecs.open(fname,"r","utf-8") as stream:
        result = stream.read()
    if nonl and len(result)>0 and result[-1]=='\n':
        result = result[:-1]
    if normalize:
        result = normalize_text(result)
    return result

def write_text(fname,text,nonl=0,normalize=1):
    """Write text. This assumes files are in unicode.
    By default, it removes newlines and normalizes the
    text for OCR processing with `normalize_text`"""
    if normalize:
        text = normalize_text(text)
    with codecs.open(fname,"w","utf-8") as stream:
        stream.write(text)
        if not nonl and text[-1]!='\n':
            stream.write('\n')

################################################################
### Image I/O
################################################################

def pil2array(im,alpha=0):
    if im.mode=="L":
        a = numpy.fromstring(im.tobytes(),'B')
        a.shape = im.size[1],im.size[0]
        return a
    if im.mode=="RGB":
        a = numpy.fromstring(im.tobytes(),'B')
        a.shape = im.size[1],im.size[0],3
        return a
    if im.mode=="RGBA":
        a = numpy.fromstring(im.tobytes(),'B')
        a.shape = im.size[1],im.size[0],4
        if not alpha: a = a[:,:,:3]
        return a
    return pil2array(im.convert("L"))

def array2pil(a):
    if a.dtype==dtype("B"):
        if a.ndim==2:
            return PIL.Image.frombytes("L",(a.shape[1],a.shape[0]),a.tostring())
        elif a.ndim==3:
            return PIL.Image.frombytes("RGB",(a.shape[1],a.shape[0]),a.tostring())
        else:
            raise OcropusException("bad image rank")
    elif a.dtype==dtype('float32'):
        return PIL.Image.fromstring("F",(a.shape[1],a.shape[0]),a.tostring())
    else:
        raise OcropusException("unknown image type")

def isbytearray(a):
    return a.dtype in [dtype('uint8')]

def isfloatarray(a):
    return a.dtype in [dtype('f'),dtype('float32'),dtype('float64')]

def isintarray(a):
    return a.dtype in [dtype('B'),dtype('int16'),dtype('int32'),dtype('int64'),dtype('uint16'),dtype('uint32'),dtype('uint64')]

def isintegerarray(a):
    return a.dtype in [dtype('int32'),dtype('int64'),dtype('uint32'),dtype('uint64')]

@checks(str,pageno=int,_=GRAYSCALE)
def read_image_gray(fname,pageno=0):
    """Read an image and returns it as a floating point array.
    The optional page number allows images from files containing multiple
    images to be addressed.  Byte and short arrays are rescaled to
    the range 0...1 (unsigned) or -1...1 (signed)."""
    if type(fname)==tuple: fname,pageno = fname
    assert pageno==0
    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    if a.dtype==dtype('uint8'):
        a = a/255.0
    if a.dtype==dtype('int8'):
        a = a/127.0
    elif a.dtype==dtype('uint16'):
        a = a/65536.0
    elif a.dtype==dtype('int16'):
        a = a/32767.0
    elif isfloatarray(a):
        pass
    else:
        raise OcropusException("unknown image type: "+a.dtype)
    if a.ndim==3:
        a = mean(a,2)
    return a


def write_image_gray(fname,image,normalize=0,verbose=0):
    """Write an image to disk.  If the image is of floating point
    type, its values are clipped to the range [0,1],
    multiplied by 255 and converted to unsigned bytes.  Otherwise,
    the image must be of type unsigned byte."""
    if verbose: print("# writing", fname)
    if isfloatarray(image):
        image = array(255*clip(image,0.0,1.0),'B')
    assert image.dtype==dtype('B'),"array has wrong dtype: %s"%image.dtype
    im = array2pil(image)
    im.save(fname)

@checks(str,_=ABINARY2)
def read_image_binary(fname,dtype='i',pageno=0):
    """Read an image from disk and return it as a binary image
    of the given dtype."""
    if type(fname)==tuple: fname,pageno = fname
    assert pageno==0
    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    if a.ndim==3: a = amax(a,axis=2)
    return array(a>0.5*(amin(a)+amax(a)),dtype)

@checks(str,ABINARY2)
def write_image_binary(fname,image,verbose=0):
    """Write a binary image to disk. This verifies first that the given image
    is, in fact, binary.  The image may be of any type, but must consist of only
    two values."""
    if verbose: print("# writing", fname)
    assert image.ndim==2
    image = array(255*(image>midrange(image)),'B')
    im = array2pil(image)
    im.save(fname)

@checks(AINT3,_=AINT2)
def rgb2int(a):
    """Converts a rank 3 array with RGB values stored in the
    last axis into a rank 2 array containing 32 bit RGB values."""
    assert a.ndim==3
    assert a.dtype==dtype('B')
    return array(0xffffff&((0x10000*a[:,:,0])|(0x100*a[:,:,1])|a[:,:,2]),'i')

@checks(AINT2,_=AINT3)
def int2rgb(image):
    """Converts a rank 3 array with RGB values stored in the
    last axis into a rank 2 array containing 32 bit RGB values."""
    assert image.ndim==2
    assert isintarray(image)
    a = zeros(list(image.shape)+[3],'B')
    a[:,:,0] = (image>>16)
    a[:,:,1] = (image>>8)
    a[:,:,2] = image
    return a

@checks(LIGHTSEG,_=DARKSEG)
def make_seg_black(image):
    assert isintegerarray(image),"%s: wrong type for segmentation"%image.dtype
    image = image.copy()
    image[image==0xffffff] = 0
    return image

@checks(DARKSEG,_=LIGHTSEG)
def make_seg_white(image):
    assert isintegerarray(image),"%s: wrong type for segmentation"%image.dtype
    image = image.copy()
    image[image==0] = 0xffffff
    return image

@checks(str,_=LINESEG)
def read_line_segmentation(fname):
    """Reads a line segmentation, that is an RGB image whose values
    encode the segmentation of a text line.  Returns an int array."""
    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    assert a.dtype==dtype('B')
    assert a.ndim==3
    image = rgb2int(a)
    result = make_seg_black(image)
    return result

@checks(str,LINESEG)
def write_line_segmentation(fname,image):
    """Writes a line segmentation, that is an RGB image whose values
    encode the segmentation of a text line."""
    a = int2rgb(make_seg_white(image))
    im = array2pil(a)
    im.save(fname)

@checks(str,_=PAGESEG)
def read_page_segmentation(fname):
    """Reads a page segmentation, that is an RGB image whose values
    encode the segmentation of a page.  Returns an int array."""
    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    assert a.dtype==dtype('B')
    assert a.ndim==3
    segmentation = rgb2int(a)
    segmentation = make_seg_black(segmentation)
    return segmentation

@checks(str,PAGESEG)
def write_page_segmentation(fname,image):
    """Writes a page segmentation, that is an RGB image whose values
    encode the segmentation of a page."""
    assert image.ndim==2
    assert image.dtype in [dtype('int32'),dtype('int64')]
    a = int2rgb(make_seg_white(image))
    im = array2pil(a)
    im.save(fname)

def iulib_page_iterator(files):
    for fname in files:
        image = read_image_gray(fname)
        yield image,fname

def norm_max(a):
    return a/amax(a)

def pad_by(image,r,dtype=None):
    """Symmetrically pad the image by the given amount.
    FIXME: replace by scipy version."""
    if dtype is None: dtype = image.dtype
    w,h = image.shape
    result = zeros((w+2*r,h+2*r))
    result[r:(w+r),r:(h+r)] = image
    return result
class RegionExtractor:
    """A class facilitating iterating over the parts of a segmentation."""
    def __init__(self):
        self.cache = {}
    def clear(self):
        del self.cache
        self.cache = {}
    def setImage(self,image):
        return self.setImageMasked(image)
    def setImageMasked(self,image,mask=None,lo=None,hi=None):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This picks a subset of the segmentation to iterate
        over, using a mask and lo and hi values.."""
        assert image.dtype==dtype('B') or image.dtype==dtype('i'),"image must be type B or i"
        if image.ndim==3: image = rgb2int(image)
        assert image.ndim==2,"wrong number of dimensions"
        self.image = image
        labels = image
        if lo is not None: labels[labels<lo] = 0
        if hi is not None: labels[labels>hi] = 0
        if mask is not None: labels = bitwise_and(labels,mask)
        labels,correspondence = morph.renumber_labels_ordered(labels,correspondence=1)
        self.labels = labels
        self.correspondence = correspondence
        self.objects = [None]+morph.find_objects(labels)
    def setPageColumns(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This iterates over the columns."""
        self.setImageMasked(image,0xff0000,hi=0x800000)
    def setPageParagraphs(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This iterates over the paragraphs (if present
        in the segmentation)."""
        self.setImageMasked(image,0xffff00,hi=0x800000)
    def setPageLines(self,image):
        """Set the image to be iterated over.  This should be an RGB image,
        ndim==3, dtype=='B'.  This iterates over the lines."""
        self.setImageMasked(image,0xffffff,hi=0x800000)
    def id(self,i):
        """Return the RGB pixel value for this segment."""
        return self.correspondence[i]
    def x0(self,i):
        """Return x0 (column) for the start of the box."""
        return self.bbox(i)[1]
    def x1(self,i):
        """Return x0 (column) for the end of the box."""
        return self.bbox(i)[3]
    def y0(self,i):
        """Return y0 (row) for the start of the box."""
        h = self.image.shape[0]
        return h-self.bbox(i)[2]-1
    def y1(self,i):
        """Return y0 (row) for the end of the box."""
        h = self.image.shape[0]
        return h-self.bbox(i)[0]-1
    def bbox(self,i):
        """Return the bounding box in raster coordinates
        (row0,col0,row1,col1)."""
        r = self.objects[i]
        # print("@@@bbox", i, r)
        return (r[0].start,r[1].start,r[0].stop,r[1].stop)
    def bboxMath(self,i):
        """Return the bounding box in math coordinates
        (row0,col0,row1,col1)."""
        h = self.image.shape[0]
        (y0,x0,y1,x1) = self.bbox(i)
        return (h-y1-1,x0,h-y0-1,x1)
    def length(self):
        """Return the number of components."""
        return len(self.objects)
    def mask(self,index,margin=0):
        """Return the mask for component index."""
        b = self.objects[index]
        # print("@@@mask", index, b)
        m = self.labels[b]
        m[m!=index] = 0
        if margin>0: m = pad_by(m,margin)
        return array(m!=0,'B')
    def extract(self,image,index,margin=0):
        """Return the subimage for component index."""
        h,w = image.shape[:2]
        (r0,c0,r1,c1) = self.bbox(index)
        # mask = self.mask(index,margin=margin)
        return image[max(0,r0-margin):min(h,r1+margin),max(0,c0-margin):min(w,c1+margin),...]
    def extractMasked(self,image,index,grow=0,bg=None,margin=0,dtype=None):
        """Return the masked subimage for component index, elsewhere the bg value."""
        if bg is None: bg = amax(image)
        h,w = image.shape[:2]
        mask = self.mask(index,margin=margin)
        # FIXME ... not circular
        if grow>0: mask = morphology.binary_dilation(mask,iterations=grow)
        mh,mw = mask.shape
        box = self.bbox(index)
        r0,c0,r1,c1 = box
        subimage = sl.cut(image,(r0,c0,r0+mh-2*margin,c0+mw-2*margin),margin,bg=bg)
        return where(mask,subimage,bg)



################################################################
### Object reading and writing
### This handles reading and writing zipped files directly,
### and it also contains workarounds for changed module/class names.
################################################################

def save_object(fname,obj,zip=0):
    if zip==0 and fname.endswith(".gz"):
        zip = 1
    if zip>0:
        # with gzip.GzipFile(fname,"wb") as stream:
        with os.popen("gzip -9 > '%s'"%fname,"wb") as stream:
            cPickle.dump(obj,stream,2)
    else:
        with open(fname,"wb") as stream:
            cPickle.dump(obj,stream,2)

def unpickle_find_global(mname,cname):
    if mname=="lstm.lstm":
        return getattr(lstm,cname)
    if not mname in sys.modules.keys():
        exec "import "+mname
    return getattr(sys.modules[mname],cname)

def load_object(fname,zip=0,nofind=0,verbose=0):
    """Loads an object from disk. By default, this handles zipped files
    and searches in the usual places for OCRopus. It also handles some
    class names that have changed."""
    if not nofind:
        fname = ocropus_find_file(fname)
    if verbose:
        print("# loading object", fname)
    if zip==0 and fname.endswith(".gz"):
        zip = 1
    if zip>0:
        # with gzip.GzipFile(fname,"rb") as stream:
        with os.popen("gunzip < '%s'"%fname,"rb") as stream:
            unpickler = cPickle.Unpickler(stream)
            unpickler.find_global = unpickle_find_global
            return unpickler.load()
    else:
        with open(fname,"rb") as stream:
            unpickler = cPickle.Unpickler(stream)
            unpickler.find_global = unpickle_find_global
            return unpickler.load()



################################################################
### Simple record object.
################################################################

class Record:
    """A simple record datatype that allows initialization with
    keyword arguments, as in Record(x=3,y=9)"""
    def __init__(self,**kw):
        self.__dict__.update(kw)
    def like(self,obj):
        self.__dict__.update(obj.__dict__)
        return self

################################################################
### Histograms
################################################################

def chist(l):
    """Simple counting histogram.  Takes a list of items
    and returns a list of (count,object) tuples."""
    counts = {}
    for c in l:
        counts[c] = counts.get(c,0)+1
    hist = [(v,k) for k,v in counts.items()]
    return sorted(hist,reverse=1)

################################################################
### multiprocessing
################################################################

def number_of_processors():
    """Estimates the number of processors."""
    return multiprocessing.cpu_count()
    # return int(os.popen("cat /proc/cpuinfo  | grep 'processor.*:' | wc -l").read())

def parallel_map(fun,jobs,parallel=0,chunksize=1):
    if parallel<2:
        for e in jobs:
            result = fun(e)
            yield result
    else:
        try:
            pool = multiprocessing.Pool(parallel)
            for e in pool.imap_unordered(fun,jobs,chunksize):
                yield e
        finally:
            pool.close()
            pool.join()
            del pool

def check_valid_class_label(s):
    """Determines whether the given character is a valid class label.
    Control characters and spaces are not permitted."""
    if type(s)==unicode:
        if re.search(r'[\0-\x20]',s):
            raise BadClassLabel(s)
    elif type(s)==str:
        if re.search(r'[^\x21-\x7e]',s):
            raise BadClassLabel(s)
    else:
        raise BadClassLabel(s)

################################################################
### file name manipulation
################################################################

@checks(str,_=str)
def findfile(name,error=1):
    result = ocropus_find_file(name)
    return result

@checks(str)
def finddir(name):
    """Find some OCRopus-related resource by looking in a bunch off standard places.
    (This needs to be integrated better with setup.py and the build system.)"""
    local = getlocal()
    path = name
    if os.path.exists(path) and os.path.isdir(path): return path
    path = local+name
    if os.path.exists(path) and os.path.isdir(path): return path
    _,tail = os.path.split(name)
    path = tail
    if os.path.exists(path) and os.path.isdir(path): return path
    path = local+tail
    if os.path.exists(path) and os.path.isdir(path): return path
    raise FileNotFound("file '"+path+"' not found in . or /usr/local/share/ocropus/")

@checks(str)
def allsplitext(path):
    """Split all the pathname extensions, so that "a/b.c.d" -> "a/b", ".c.d" """
    match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
    if not match:
        return path,""
    else:
        return match.group(1),match.group(3)

@checks(str)
def base(path):
    return allsplitext(path)[0]

@checks(str,{str,unicode})
def write_text_simple(file,s):
    """Write the given string s to the output file."""
    with open(file,"w") as stream:
        if type(s)==unicode: s = s.encode("utf-8")
        stream.write(s)

@checks([str])
def glob_all(args):
    """Given a list of command line arguments, expand all of them with glob."""
    result = []
    for arg in args:
        if arg[0]=="@":
            with open(arg[1:],"r") as stream:
                expanded = stream.read().split("\n")
            expanded = [s for s in expanded if s!=""]
        else:
            expanded = sorted(glob.glob(arg))
        if len(expanded)<1:
            raise FileNotFound("%s: expansion did not yield any files"%arg)
        result += expanded
    return result

@checks([str])
def expand_args(args):
    """Given a list of command line arguments, if the
    length is one, assume it's a book directory and expands it.
    Otherwise returns the arguments unchanged."""
    if len(args)==1 and os.path.isdir(args[0]):
        return sorted(glob.glob(args[0]+"/????/??????.png"))
    else:
        return args


def ocropus_find_file(fname, gz=True):
    """Search for `fname` in one of the OCRopus data directories, as well as
    the current directory). If `gz` is True, search also for gzipped files.

    Result of searching $fname is the first existing in:

        * $base/$fname
        * $base/$fname.gz       # if gz
        * $base/model/$fname
        * $base/model/$fname.gz # if gz
        * $base/data/$fname
        * $base/data/$fname.gz  # if gz
        * $base/gui/$fname
        * $base/gui/$fname.gz   # if gz

    $base can be four base paths:
        * `$OCROPUS_DATA` environment variable
        * current working directory
        * ../../../../share/ocropus from this file's install location
        * `/usr/local/share/ocropus`
        * `$PREFIX/share/ocropus` ($PREFIX being the Python installation 
           prefix, usually `/usr`)
    """
    possible_prefixes = []

    if os.getenv("OCROPUS_DATA"):
        possible_prefixes.append(os.getenv("OCROPUS_DATA"))

    possible_prefixes.append(os.curdir)

    possible_prefixes.append(os.path.normpath(os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        os.pardir, os.pardir, os.pardir, os.pardir, "share", "ocropus")))

    possible_prefixes.append("/usr/local/share/ocropus")

    possible_prefixes.append(os.path.join(
        sysconfig.get_config_var("datarootdir"), "ocropus"))


    # Unique entries with preserved order in possible_prefixes
    # http://stackoverflow.com/a/15637398/201318
    possible_prefixes = [possible_prefixes[i] for i in
            sorted(numpy.unique(possible_prefixes, return_index=True)[1])]
    for prefix in possible_prefixes:
        if not os.path.isdir(prefix):
            continue
        for basename in [".", "models", "data", "gui"]:
            if not os.path.isdir(os.path.join(prefix, basename)):
                continue
            full = os.path.join(prefix, basename, fname)
            if os.path.exists(full):
                return full
            if gz and os.path.exists(full + ".gz"):
                return full + ".gz"

    raise FileNotFound(fname)


def fvariant(fname,kind,gt=""):
    """Find the file variant corresponding to the given file name.
    Possible fil variants are line (or png), rseg, cseg, fst, costs, and txt.
    Ground truth files have an extra suffix (usually something like "gt",
    as in 010001.gt.txt or 010001.rseg.gt.png).  By default, the variant
    with the same ground truth suffix is produced.  The non-ground-truth
    version can be produced with gt="", the ground truth version can
    be produced with gt="gt" (or some other desired suffix)."""
    if gt!="": gt = "."+gt
    base,ext = allsplitext(fname)
    # text output
    if kind=="txt":
        return base+gt+".txt"
    assert gt=="","gt suffix may only be supplied for .txt files (%s,%s,%s)"%(fname,kind,gt)
    # a text line image
    if kind=="line" or kind=="png" or kind=="bin":
        return base+".bin.png"
    if kind=="nrm":
        return base+".nrm.png"
    # a recognition lattice
    if kind=="lattice":
        return base+gt+".lattice"
    # raw segmentation
    if kind=="rseg":
        return base+".rseg.png"
    # character segmentation
    if kind=="cseg":
        return base+".cseg.png"
    # text specifically aligned with cseg (this may be different from gt or txt)
    if kind=="aligned":
        return base+".aligned"
    # per character costs
    if kind=="costs":
        return base+".costs"
    raise BadInput("unknown kind: %s"%kind)

################################################################
### Utility for setting "parameters" on an object: a list of keywords for
### changing instance variables.
################################################################

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

################################################################
### warning and logging
################################################################

def caller():
    """Just returns info about the caller in string for (for error messages)."""
    frame = sys._getframe(2)
    info = inspect.getframeinfo(frame)
    result = "%s:%d (%s)"%(info.filename,info.lineno,info.function)
    del frame
    return result

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
        print("import", module)
        exec "import "+module in env
    return eval(s,env)

def mkpython(name):
    """Tries to instantiate a Python class.  Gives an error if it looks
    like a Python class but can't be instantiated.  Returns None if it
    doesn't look like a Python class."""
    if name is None or len(name)==0:
        return None
    elif type(name) is not str:
        return name()
    elif name[0]=="=":
        return pyconstruct(name[1:])
    elif "(" in name or "." in name:
        return pyconstruct(name)
    else:
        return None

################################################################
### loading and saving components
################################################################

# This code has to deal with a lot of special cases for all the
# different formats we have accrued.

def obinfo(ob):
    """A bit of information about the given object.  Returns
    the str representation of the object, and if it has a shape,
    also includes the shape."""
    result = str(ob)
    if hasattr(ob,"shape"):
        result += " "
        result += str(ob.shape)
    return result


def binarize_range(image,dtype='B',threshold=0.5):
    """Binarize an image by its range."""
    threshold = (amax(image)+amin(image))*threshold
    scale = 1
    if dtype=='B': scale = 255
    return array(scale*(image>threshold),dtype=dtype)

def plotgrid(data,d=10,shape=(30,30)):
    """Plot a list of images on a grid."""
    ion()
    gray()
    clf()
    for i in range(min(d*d,len(data))):
        subplot(d,d,i+1)
        row = data[i]
        if shape is not None: row = row.reshape(shape)
        imshow(row)
    ginput(1,timeout=0.1)

def showrgb(r,g=None,b=None):
    if g is None: g = r
    if b is None: b = r
    imshow(array([r,g,b]).transpose([1,2,0]))


def gt_explode(s):
    l = re.split(r'_(.{1,4})_',s)
    result = []
    for i,e in enumerate(l):
        if i%2==0:
            result += [c for c in e]
        else:
            result += [e]
    result = [re.sub("\001","_",s) for s in result]
    result = [re.sub("\002","\\\\",s) for s in result]
    return result

def gt_implode(l):
    result = []
    for c in l:
        if c=="_":
            result.append("___")
        elif len(c)<=1:
            result.append(c)
        elif len(c)<=4:
            result.append("_"+c+"_")
        else:
            raise BadInput("cannot create ground truth transcription for: %s"%l)
    return "".join(result)

@checks(int,sequence=int,frac=int,_=BOOL)
def testset(index,sequence=0,frac=10):
    # this doesn't have to be good, just a fast, somewhat random function
    return sequence==int(abs(sin(index))*1.23456789e6)%frac

def midrange(image,frac=0.5):
    """Computes the center of the range of image values
    (for quick thresholding)."""
    return frac*(amin(image)+amax(image))

def remove_noise(line,minsize=8):
    """Remove small pixels from an image."""
    if minsize==0: return line
    bin = (line>0.5*amax(line))
    labels,n = morph.label(bin)
    sums = measurements.sum(bin,labels,range(n+1))
    sums = sums[labels]
    good = minimum(bin,1-(sums>0)*(sums<minsize))
    return good

class MovingStats:
    def __init__(self,n=100):
        self.data = []
        self.n = n
        self.count = 0
    def add(self,x):
        self.data += [x]
        self.data = self.data[-self.n:]
        self.count += 1
    def mean(self):
        if len(self.data)==0: return nan
        return mean(self.data)
