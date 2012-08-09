import collections
import functools
import linecache
import numbers
import numpy
import os
import sys
import traceback
import warnings
from types import NoneType

### printing

def strc(arg,n=10):
    """Compact version of `str`."""
    if isinstance(arg,float):
        return "%.3g"%arg
    if type(arg)==list:
        return "[%s|%d]"%(",".join([strc(x) for x in arg[:3]]),len(arg))
    if type(arg)==numpy.ndarray:
        return "<ndarray-%x %s %s [%s,%s]>"%(id(arg),arg.shape,str(arg.dtype),numpy.amin(arg),numpy.amax(arg))
    return str(arg).replace("\n"," ")

### deprecation warnings

def deprecated(f):
    """Prints a deprecation warning when called."""
    @functools.wraps(f)
    def wrapper(*args,**kw):
        warnings.warn_explicit("calling deprecated function %s"%f.__name__,
                               category=DeprecationWarning,
                               filename=f.func_code.co_filename,
                               lineno=f.func_code.co_firstlineno+1)
        return f(*args,**kw)
    return wrapper

def failfunc(f):
    @functools.wraps(f)
    def wrapper(*args,**kw):
        raise Exception("don't call %s anymore"%f)
    return wrapper

obsolete = failfunc

### debugging / tracing

_trace1_depth = 0

def trace1(f):
    """Print arguments/return values for the decorated function before each call."""
    name = f.func_name
    argnames = f.func_code.co_varnames[:f.func_code.co_argcount]
    @functools.wraps(f)
    def wrapper(*args,**kw):
        try:
            global _trace1_depth
            _trace1_depth += 1
            print " "*_trace1_depth,"ENTER",name,":",
            for k,v in zip(argnames,args)+kw.items():
                print "%s=%s"%(k,strc(v)),
            print
            result = f(*args,**kw)
            print " "*_trace1_depth,"LEAVE",name,":",strc(result)
            return result
        except Exception,e:
            print " "*_trace1_depth,"ERROR",name,":",e
        finally:
            _trace1_depth -= 1
    return wrapper

def tracing(f):
    """Enable tracing just within a function call."""
    def globaltrace(frame,why,arg):
        if why == "call": return localtrace
        return None
    def localtrace(frame, why, arg):
        if why == "line":
            fname = frame.f_code.co_filename
            lineno = frame.f_lineno
            base = os.path.basename(fname)
            print "%s(%s): %s"%(base,lineno,linecache.getline(fname,lineno))
        return localtrace
    @wrap(f)
    def wrapper(*args,**kw):
        sys.settrace(globaltrace)
        result = f(*args,**kw)
        sys.settrace(None)
        return result
    return wrapper

def method(cls):
    """Adds the function as a method to the given class."""
    import new
    def _wrap(f):
        cls.__dict__[f.func_name] = new.instancemethod(f,None,cls)
        return None
    return _wrap

def unchanged(f):
    "This decorator doesn't add any behavior"
    return f

def disabled(value=None):
    """Disables the function so that it does nothing.  Optionally
    returns the given value."""
    def wrapper(f):
        @functools.wraps(f)
        def g(*args,**kw):
            return value
        return g
    return wrapper

def replacedby(g):
    """Replace the function with another function."""
    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*args,**kw):
            return g(*args,**kw)
        return wrapped
    return wrapper

### type and range checks for arguments and return values

class CheckError(Exception):
    def __init__(self,*args,**kw):
        self.description = " ".join([strc(x) for x in args])
        self.kw = kw
    def __str__(self):
        result = "<CheckError %s"%self.description
        for k,v in self.kw.items():
            result += " %s=%s"%(k,strc(v))
        result += ">"
        return result

def BOOL(x):
    if isinstance(x,bool): return 1
    if isinstance(x,int) and x in [0,1]: return 1
    return 0
def NUMBER(a):
    return isinstance(a,int) or isinstance(a,float)
class RANGE:
    def __init__(self,lo,hi):
        self.lo = lo
        self.hi = hi
    def __str__(self):
        return "[%g,%g]"%(self.lo,self.hi)

def checks(*types,**ktypes):
    def decorator(f):
        def check(var,value,type_):
            # None skips any check
            if type_ is True:
                return 1
            # types are checked as types
            if type(type_)==type:
                if not isinstance(value,type_):
                    raise CheckError("isinstance failed",value,type_,var=var)
                return 1
            # for a list, check that all elements of a collection have a type
            # of some list element, allowing declarations like [str] or [str,unicode]
            # no recursive checks right now
            if type(type_)==list:
                if not numpy.iterable(value):
                    raise CheckError("expected iterable",value,var=var)
                for x in value:
                    if not reduce(max,[isinstance(x,t) for t in type_]):
                        raise CheckError("element",x,"fails to be of type",type_,var=var)
                return 1
            # for sets, check membership of the type in the set
            if type(type_)==set:
                for t in type_:
                    if isinstance(value,t): return 1
                raise CheckError("set membership failed",value,type_,var=var)
            # lists of length two are checked as intervals
            if isinstance(type_,RANGE):
                if not NUMBER(value): raise CheckError("range check failed, expected a number",value,var=var)
                if value<type_.lo or value>type_.hi: raise CheckError("range check failed",value,"wanted",type_,var=var)
                return 1
            # callables are checked as assertions
            if callable(type_):
                # callables can also raise the type error themselves
                kw = dict()
                if "var" in type_.func_code.co_varnames: kw = dict(var=var)
                if not type_(value,**kw): raise CheckError("check failed",value,type_,var=var)
                return 1
            # otherwise, we don't understand the type spec
            raise Exception("unknown type spec: %s"%type_)
        @functools.wraps(f)
        def wrapper(*args,**kw):
            # print "@@@",f,"decl",types,ktypes,"call",[strc(x) for x in args],kw
            name = f.func_name
            argnames = f.func_code.co_varnames[:f.func_code.co_argcount]
            kw3 = [(var,value,ktypes.get(var,None)) for var,value in kw.items()]
            for var,value,type in zip(argnames,args,types)+kw3:
                check(var,value,type)
            result = f(*args,**kw)
            check("return",result,kw.get("_",True))
            return result
        return wrapper
    return decorator

def NDARRAY(atype=None,ndim=None,range=None):
    if type(ndim)==tuple:
        assert len(ndim)==2
        ndim_lo = ndim[0]
        ndim_hi = ndim[1]
    elif ndim is None:
        ndim_lo = 1
        ndim_hi = 16
    else:
        assert type(ndim)==int
        ndim_lo = ndim
        ndim_hi = ndim
    assert range is None or (type(range)==list and len(range)==2),"bad range specification, use lists"
    def check(a,var=None):
        if not isinstance(a,numpy.ndarray):
            raise CheckError("ndarray",a,var)
        if a.ndim<ndim_lo or a.ndim>ndim_hi:
            if ndim_lo==ndim_hi:
                raise CheckError("ndarray",a,"ndim",a.ndim,"!=",ndim_lo,var=var)
            else:
                raise CheckError("ndarray",a,"ndim",a.ndim,"not in range",ndim_lo,ndim_hi,var=var)
        if isinstance(atype,numpy.dtype):
            if atype!=dtype:
                raise CheckError("ndarray",a,"dtype",a.dtype,"not",dtype,var=var)
        if atype==float:
            if a.dtype not in [numpy.dtype('float32'),numpy.dtype('float64'),numpy.dtype('float128')]:
                raise CheckError("ndarray",a,"dtype",a.dtype,"not a float type",var=var)
        if atype==int:
            if a.dtype not in [numpy.dtype('uint8'),numpy.dtype('int32'),numpy.dtype('int64'),numpy.dtype('uint32')]:
                raise CheckError("ndarray",a,"dtype",a.dtype,"not an int type",var=var)
        if range is not None:
            arange = (numpy.amin(a),numpy.amax(a))
            if numpy.amin(a)<range[0] or numpy.amax(a)>range[1]:
                raise CheckError("ndarray",a,"range",arange,"outside",tuple(range),var=var)
        return 1
    return check

def inttuple(a):
    if isinstance(a,int): return 1
    if not (tuple(a) or list(a)): return 0
    for x in a:
        if not isinstance(x,int): return 0
    return 1
def uinttuple(a):
    if isinstance(a,int): return 1
    if not (tuple(a) or list(a)): return 0
    for x in a:
        if not isinstance(x,int): return 0
        if x<0: return 0
    return 1
def uintpair(a):
    if not tuple(a): return 0
    if not len(a)==2: return 0
    if a[0]<0: return 0
    if a[1]<0: return 0
    return 1

def RECTANGLE(a):
    if not tuple(a): return 0
    if not isinstance(a[0],slice): return 0
    if not isinstance(a[1],slice): return 0
    return 1

### specific kinds of arrays

ARRAY1 = NDARRAY(ndim=1)
ARRAY2 = NDARRAY(ndim=2)
ARRAY3 = NDARRAY(ndim=3)
INT1 = NDARRAY(ndim=1,atype=int)
INT2 = NDARRAY(ndim=2,atype=int)
INT3 = NDARRAY(ndim=3,atype=int)
FLOAT1 = NDARRAY(ndim=1,atype=float)
FLOAT2 = NDARRAY(ndim=2,atype=float)
FLOAT3 = NDARRAY(ndim=3,atype=float)

def BINARY(a):
    if a.ndim==2 and a.dtype==numpy.dtype(bool): return 1
    if not INT2(a): return 0
    import scipy.ndimage.measurements
    zeros,ones = scipy.ndimage.measurements.sum(1,a,[0,1])
    if not zeros+ones == a.size: return 0
    return 1
def BINARY1(a):
    return ARRAY1(a) and BINARY(a)
def BINARY2(a):
    return ARRAY2(a) and BINARY(a)
def BINARY3(a):
    return ARRAY3(a) and BINARY(a)
def GRAYSCALE(a):
    return isinstance(a,numpy.ndarray) and a.ndim==2 and isinstance(a.flat[0],numbers.Number)
def RGB(a):
    return isinstance(a,numpy.ndarray) and a.ndim==3 and a.shape[2]==3 and isinstance(a.flat[0],numbers.Number)
def RGBA(a):
    return isinstance(a,numpy.ndarray) and a.ndim==3 and a.shape[2]==4 and isinstance(a.flat[0],numbers.Number)

### arrays with range checks

def GRAYSCALE1(a):
    return isinstance(a,numpy.ndarray) and a.ndim==2 and isinstance(a.flat[0],float) and numpy.amin(a)>=0 and numpy.amax(a)<=1
def RGB1(a):
    return isinstance(a,numpy.ndarray) and a.ndim==3 and a.shape[2]==3 and isinstance(a.flat[0],float) and numpy.amin(a)>=0 and numpy.amax(a)<=1
def RGBA1(a):
    return isinstance(a,numpy.ndarray) and a.ndim==3 and a.shape[2]==4 and isinstance(a.flat[0],float) and numpy.amin(a)>=0 and numpy.amax(a)<=1

### image arrays with more complicated image properties

def LIGHT(a,var=None):
    if not FLOAT2(a):
        raise CheckError("image",a,"must be float type",var=var)
    if numpy.amin(a)<0:
        raise CheckError("image",a,"must be non-negative",a,var=var)
    if numpy.median(a)<=numpy.mean(a):
        raise CheckError("image",a,"is not 'light'; median",numpy.median(a),"mean",numpy.mean(a),var=var)
    return 1
def DARK(a,var=None):
    if not FLOAT2(a):
        raise CheckError("image",a,"must be float type",var=var)
    if numpy.amin(a)<0:
        raise CheckError("image",a,"must be non-negative",a,var=var)
    if numpy.median(a)>=numpy.mean(a):
        raise CheckError("image",a,"is not 'dark'; median",numpy.median(a),"mean",numpy.mean(a),var=var)
    return 1
def PAGE(a,var=None):
    if not NDARRAY(rank=2)(a): 
        raise CheckError("page image",a,"must be a 2D array",a,var=var)
    if not (a.ndim==2 and a.shape[0]>300 and a.shape[1]>300):
        raise CheckError("page image",a,"has the wrong shape",a.shape)
    return 1
def LINE(a,var=None):
    if not NDARRAY(rank=2)(a): 
        raise CheckError("line image",l,"must be a 2D array",a,var=var)
    if not a.shape[0]>5 and a.shape[1]>20 and a.shape[0]<a.shape[1]:
        raise CheckError("line image",l,"has the wrong shape",a.shape)
    return 1
def BINPAGE(a):
    return PAGE(a) and BINARY(a)
def LIGHTPAGE(a):
    return PAGE(a) and LIGHT(a)
def DARKPAGE(a):
    return PAGE(a) and DARK(a)
def LIGHTLINE(a):
    return LINE(a) and LIGHT(a)
def DARKLINE(a):
    return LINE(a) and DARK(a)
def PATCH(a):
    return GRAYSCALE1(a) and a.shape[0]<200 and a.shape[1]<200

### segmentation-related checks
###
### Segmentations come in two flavors: with a white background (for writing to disk
### so that one can see something in file browsers), and with a black background
### (for easy processing). Light segmentations should only exist on disk.

def SEGMENTATION(a):
    return isinstance(a,numpy.ndarray) and a.ndim==2 and a.dtype in ['int32','int64']
def LIGHTSEG(a):
    return SEGMENTATION(a) and numpy.amax(a)==0xffffff and numpy.amin(a)==1
def DARKSEG(a):
    return SEGMENTATION(a) and numpy.amax(a)<0xffffff and numpy.amin(a)==0
def PAGESEG(a):
    return DARKSEG(a) and PAGE(a)
def LINESEG(a):
    return DARKSEG(a) and LINE(a)
def LIGHTPAGESEG(a):
    return LIGHTSEG(a) and PAGE(a)
def LIGHTLINESEG(a):
    return LIGHTSEG(a) and LINE(a)

### special types for pattern recognition

def DATASET(minsize=3,maxsize=int(1e9),mindim=2,maxdim=10000,bounds=(-1000.0,1000.0),rank=None,fixedshape=0,shape=None):
    if shape is not None:
        fixedshape = 1
    def checker(dataset,var=None):
        if not hasattr(dataset,"__len__"):
            raise CheckError("dataset",dataset,"lacks __len__",var=var)
        if not hasattr(dataset,"__getitem__"):
            raise CheckError("dataset",dataset,"lacks __getitem__",var=var)
        if len(dataset)<minsize:
            raise CheckError("dataset",dataset,"too small",len(dataset),"<",minsize,var=var)
        if maxsize is not None and len(dataset)>maxsize:
            raise CheckError("dataset",dataset,"too large",len(dataset),">",maxsize,var=var)
        samples = sorted(pyrandom.sample(xrange(len(dataset)),10))
        for i in samples:
            sample = dataset[i]
            if fixedshape:
                if shape is None:
                    shape = sample.shape
                    test_index = i
                if shape!=sample.shape and shape=="fixed":
                    raise CheckError("dataset[%d].shape"%i,sample.shape,"differs from dataset[%d].shape"%test_index,shape)
                else:
                    raise CheckError("dataset[%d].shape"%i,sample.shape,"differs from expected shape",shape)
            if not NDARRAY(atype=float)(sample):
                raise CheckError("dataset[%d]"%i,dataset[i],"failed to return a float array",var=var)
            if sample.size<mindim:
                raise CheckError("dataset[%d]"%i,dataset[i],"has fewer than",mindim,"elements")
            if sample.size>maxdim:
                raise CheckError("dataset[%d]"%i,dataset[i],"has more than",mindim,"elements")
            lo,hi = numpy.amin(sample),numpy.amax(sample)
            if lo<bounds[0] or hi>bounds[1]:
                raise CheckError("dataset[%d]"%i,dataset[i],"contains values",(lo,hi),"outside the required bounds",bounds)
            if rank is not None and sample.ndim!=rank:
                raise CheckError("dataset[%d]"%i,dataset[i],"has",sample.ndim,"dimensions, want",rank)
        return 1
        
            
# class fprofile:
#     ... execution time profiling for specific functions ...

# def typecheck(f):
#     ... type checking ...
# @typecheck(int,float,(0,17),NONNEGATIVE,NARRAY2,...)
# def myfunc(...):
#     ...

