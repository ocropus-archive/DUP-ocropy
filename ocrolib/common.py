################################################################
### common functions for data structures, file name manipulation, etc.
################################################################

import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology

import improc
import docproc
import ligatures
import fstutils
import openfst
import segrec
import ocrofst
import ocrorast
import ocrolseg
import ocropreproc
import sl

import cPickle as pickle
pickle_mode = 2



################################################################
### Iterate through the regions of a color image.
################################################################

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

def renumber_labels(a):
    return renumber_labels_ordered(a)

def pyargsort(seq,cmp=cmp,key=lambda x:x):
    """Like numpy's argsort, but using the builtin Python sorting
    function.  Takes an optional cmp."""
    return sorted(range(len(seq)),key=lambda x:key(seq.__getitem__(x)),cmp=cmp)

def renumber_labels_by_boxes(a,cmp=cmp,key=lambda x:x,correspondence=0):
    """Renumber the labels of the input array according to some
    order on their bounding boxes.  If you provide a cmp function,
    it is passed the outputs of find_objects for sorting.
    The default is lexicographic."""
    if cmp=='rlex':
        import __builtin__
        cmp = lambda x,y: __builtin__.cmp(x[::-1],y[::-1])
    assert a.dtype==dtype('B') or a.dtype==dtype('i')
    labels = renumber_labels_ordered(a)
    objects = flexible_find_objects(labels)
    order = array(pyargsort(objects,cmp=cmp,key=key),'i')
    assert len(objects)==len(order)
    order = concatenate(([0],order+1))
    if correspondence:
        return order[labels],argsort(order)
    else:
        return order[labels]

def flexible_find_objects(image):
    # first try the default type
    try: return measurements.find_objects(image)
    except: pass
    types = ["int32","int64","int16"]
    for t in types:
        # try with type conversions
	try: return measurements.find_objects(array(image,dtype=t)) 
	except: pass
    # let it raise the same exception as before
    return measurements.find_objects(image)

def rgb2int(image):
    assert image.dtype==dtype('B')
    orig = image
    image = zeros(image.shape[:2],'i')
    image += orig[:,:,0]
    image <<= 8
    image += orig[:,:,1]
    image <<= 8
    image += orig[:,:,2]
    return image

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
        assert image.dtype==dtype('B') or image.dtype('i'),"image must be type B or i"
        if image.ndim==3: image = rgb2int(image)
        assert image.ndim==2,"wrong number of dimensions"
        self.image = image
        labels = image
        if lo is not None: labels[labels<lo] = 0
        if hi is not None: labels[labels>hi] = 0
        if mask is not None: labels = bitwise_and(labels,mask)
        labels,correspondence = renumber_labels_ordered(labels,correspondence=1)
        self.labels = labels
        self.correspondence = correspondence
        self.objects = [None]+flexible_find_objects(labels)
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
        r = self.objects[i]
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
        m = self.labels[b]
        m[m!=index] = 0
        if margin>0: m = improc.pad_by(m,margin)
        return array(m!=0,'B')
    def extract(self,image,index,margin=0):
        """Return the subimage for component index."""
        h,w = image.shape[:2]
        (r0,c0,r1,c1) = self.bbox(index)
        mask = self.mask(index,margin=margin)
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
        subimage = improc.cut(image,(r0,c0,r0+mh-2*margin,c0+mw-2*margin),margin,bg=bg)
        return where(mask,subimage,bg)

    

################################################################
### Simple record object.
################################################################

class Record:
    def __init__(self,**kw):
        self.__dict__.update(kw)
    def like(self,obj):
        self.__dict__.update(obj.__dict__)
        return self

################################################################
### Histograms
################################################################

def chist(l):
    counts = {}
    for c in l:
        counts[c] = counts.get(c,0)+1
    hist = [(v,k) for k,v in counts.items()]
    return sorted(hist,reverse=1)

################################################################
### Environment functions
################################################################

def number_of_processors():
    try:
        return int(os.popen("cat /proc/cpuinfo  | grep 'processor.*:' | wc -l").read())
    except:
        return 1

################################################################
### exceptions
################################################################

class Unimplemented():
    def __init__(self,s):
        Exception.__init__(self,inspect.stack()[1][3])

class BadClassLabel(Exception):
    def __init__(self,s):
        Exception.__init__(self,s)

class RecognitionError(Exception):
    def __init__(self,explanation,**kw):
        self.context = kw
        s = [explanation]
        s += ["%s=%s"%(k,summary(kw[k])) for k in kw]
        message = " ".join(s)
        Exception.__init__(self,message)

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

################################################################
### file name manipulation
################################################################

def getlocal():
    local = os.getenv("OCROPUS_DATA") or "/usr/local/share/ocropus/"
    return local

def findfile(name):
    """Find some OCRopus-related resource by looking in a bunch off standard places.
    (FIXME: The implementation is pretty adhoc for now.
    This needs to be integrated better with setup.py and the build system.)"""
    local = getlocal()
    path = name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+"/gui/"+name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+"/models/"+name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+"/words/"+name
    if os.path.exists(path) and os.path.isfile(path): return path
    _,tail = os.path.split(name)
    path = tail
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+tail
    if os.path.exists(path) and os.path.isfile(path): return path
    raise IOError("file '"+path+"' not found in . or /usr/local/share/ocropus/")

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
    raise IOError("file '"+path+"' not found in . or /usr/local/share/ocropus/")

def allsplitext(path):
    """Split all the pathname extensions, so that "a/b.c.d" -> "a/b", ".c.d" """
    match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
    if not match:
        return path,""
    else:
        return match.group(1),match.group(3)

def write_text(file,s):
    """Write the given string s to the output file."""
    with open(file,"w") as stream:
        if type(s)==unicode: s = s.encode("utf-8")
        stream.write(s)

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
    if kind=="aligned":
        return base+".aligned"+gt+".txt"
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

def make_ICleanupGray(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create CleanupGray component for '%s'"%name
    assert "cleanup_gray" in dir(result)
    return result
def make_ICleanupBinary(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create CleanupBinary component for '%s'"%name
    assert "cleanup_binary" in dir(result)
    return result
def make_IBinarize(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create Binarize component for '%s'"%name
    assert "binarize" in dir(result)
    return result
def make_ITextImageClassification(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create TextImageClassification component for '%s'"%name
    assert "textImageProbabilities" in dir(result)
    return result
def make_ISegmentPage(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create SegmentPage component for '%s'"%name
    assert "segment" in dir(result)
    return result
def make_ISegmentLine(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create SegmentLine component for '%s'"%name
    assert "charseg" in dir(result)
    return result
def make_IGrouper(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create Grouper component for '%s'"%name
    assert "setSegmentation" in dir(result)
    assert "getLattice" in dir(result)
    return result
def make_IRecognizeLine(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create RecognizeLine component for '%s'"%name
    assert "recognizeLine" in dir(result)
    return result
def make_IModel(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create Model component for '%s'"%name
    assert "outputs" in dir(result)
    return result
def make_IExtractor(name):
    """Make a native component or a Python component.  Anything containing
    a "(" is assumed to be a Python component."""
    result = mkpython(name)
    assert result is not None,"cannot create Extractor component for: '%s'"%name
    assert "extract" in dir(name)
    return result

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

def rect_union(rectangles):
    if len(rectangles)<1: return (0,0,-1,-1)
    r = array(rectangles)
    return (amin(r[:,0]),amax(r[:,0]),amin(r[:,1]),amax(r[:1]))

def recognize_and_align(image,linerec,lmodel,beam=1000,nocseg=0,lig=ligatures.lig):
    """Perform line recognition with the given line recognizer and
    language model.  Outputs an object containing the result (as a
    Python string), the costs, the rseg, the cseg, the lattice and the
    total cost.  The recognition lattice needs to have rseg's segment
    numbers as inputs (pairs of 16 bit numbers); SimpleGrouper
    produces such lattices.  cseg==None means that the connected
    component renumbering failed for some reason."""

    lattice,rseg = linerec.recognizeLineSeg(image)
    v1,v2,ins,outs,costs = ocrofst.beam_search(lattice,lmodel,beam)
    result = compute_alignment(lattice,rseg,lmodel,beam=beam,lig=lig)
    return result

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
    if object.__class__.__name__=="CommonComponent" and hasattr(object,"comp"):
        # FIXME -- get rid of this eventually
        import ocropus
        ocropus.save_component(file,object.comp)
        return
    if type(object).__module__=="ocropus":
        import ocropus
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
    with open(file,"r") as stream:
        # FIXME -- get rid of this eventually
        start = stream.read(128)
    if start.startswith("<object>\nlinerec\n"):
        # FIXME -- get rid of this eventually
        warnings.warn("loading old-style linerec: %s"%file)
        result = RecognizeLine()
        import ocropus
        result.comp = ocropus.load_IRecognizeLine(file)
        return result
    if start.startswith("<object>"):
        # FIXME -- get rid of this eventually
        warnings.warn("loading old-style cmodel: %s"%file)
        import ocroold
        result = ocroold.Model()
        import ocropus
        result.comp = ocropus.load_IModel(file)
        return result
    with open(file,"rb") as stream:
        return pickle.load(stream)

def load_linerec(file,wrapper=None):
    if wrapper is None:
        wrapper=segrec.CmodelLineRecognizer
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

def simple_classify(model,inputs):
    result = []
    for i in range(len(inputs)):
        result.append(model.coutputs(inputs[i]))
    return result

