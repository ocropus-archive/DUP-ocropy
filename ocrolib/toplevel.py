from __future__ import absolute_import, division, print_function

import functools
import linecache
import numpy
import os
import sys
import warnings
from types import NoneType
# FIXME from ... import wrap

### printing

def strc(arg,n=10):
    """Compact version of `str`."""
    if isinstance(arg,float):
        return "%.3g"%arg
    if isinstance(arg, list):
        return "[%s|%d]"%(",".join([strc(x) for x in arg[:3]]),len(arg))
    if isinstance(arg, numpy.ndarray):
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
            print(" " * _trace1_depth, "ENTER", name, ":", end=' ')
            for k,v in list(zip(argnames,args))+list(kw.items()):
                print("%s=%s" % (k, strc(v)), end=' ')
            print()
            result = f(*args,**kw)
            print(" " * _trace1_depth, "LEAVE", name, ":", strc(result))
            return result
        except Exception as e:
            print(" " * _trace1_depth, "ERROR", name, ":", e)
            raise
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
            print("%s(%s): %s" % (base, lineno,
                                  linecache.getline(fname, lineno)))
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
        self.fun = kw.get("fun","?")
        self.var = kw.get("var","?")
        self.description = " ".join([strc(x) for x in args])
    def __str__(self):
        result = "\nCheckError for argument "
        result += str(self.var)
        result += " of function "
        result += str(self.fun)
        result += "\n"
        result += self.description
        return result

class CheckWarning(CheckError):
    def __init__(self,*args,**kw):
        self.fun = kw.get("fun","?")
        self.var = kw.get("var","?")
        self.description = " ".join([strc(x) for x in args])
    def __str__(self):
        result = "\nCheckWarning for argument "
        result += str(self.var)
        result += " of function "
        result += str(self.fun)
        result += "\n"
        result += self.description
        result += "(This can happen occasionally during normal operations and isn't necessarily a bug or problem.)\n"
        return result

def checktype(value,type_):
    """Check value against the type spec.  If everything
    is OK, this just returns the value itself.
    If the types don't check out, an exception is thrown."""
    # True skips any check
    if type_ is True:
        return value
    # types are checked using isinstance
    if isinstance(type_, type):
        if not isinstance(value,type_):
            raise CheckError("isinstance failed",value,"of type",type(value),"is not of type",type_)
        return value
    # for a list, check that all elements of a collection have a type
    # of some list element, allowing declarations like [str] or [str,unicode]
    # no recursive checks right now
    if isinstance(type_, list):
        if not numpy.iterable(value):
            raise CheckError("expected iterable",value)
        for x in value:
            if not any(isinstance(x, t) for t in type_):
                raise CheckError("element",x,"of type",type(x),"fails to be of type",type_)
        return value
    # for sets, check membership of the type in the set
    if isinstance(type_, set):
        for t in type_:
            if isinstance(value,t): return value
        raise CheckError("set membership failed",value,type_,var=var) # FIXME var?
    # for tuples, check that all conditions are satisfied
    if isinstance(type_, tuple):
        for t in type_:
            checktype(value,type_)
        return value
    # callables are just called and should either use assertions or
    # explicitly raise CheckError
    if callable(type_):
        type_(value)
        return value
    # otherwise, we don't understand the type spec
    raise Exception("unknown type spec: %s"%type_)

def checks(*types,**ktypes):
    """Check argument and return types against type specs at runtime."""
    def argument_check_decorator(f):
        @functools.wraps(f)
        def argument_checks(*args,**kw):
            # print("@@@", f, "decl", types, ktypes, "call",
            #       [strc(x) for x in args], kw)
            name = f.func_name
            argnames = f.func_code.co_varnames[:f.func_code.co_argcount]
            kw3 = [(var,value,ktypes.get(var,True)) for var,value in kw.items()]
            for var,value,type_ in list(zip(argnames,args,types))+kw3:
                try:
                    checktype(value,type_)
                except AssertionError as e:
                    raise CheckError(e.message,*e.args,var=var,fun=f)
                except CheckError as e:
                    e.fun = f
                    e.var = var
                    raise e
                except:
                    print("unknown exception while checking function:", name)
                    raise
            result = f(*args,**kw)
            checktype(result,kw.get("_",True))
            return result
        return argument_checks
    return argument_check_decorator

def makeargcheck(message,warning=0):
    """Converts a predicate into an argcheck."""
    def decorator(f):
        def wrapper(arg):
            if not f(arg):
                if warning:
                    raise CheckWarning(strc(arg)+" of type "+str(type(arg))+": "+str(message))
                else:
                    raise CheckError(strc(arg)+" of type "+str(type(arg))+": "+str(message))
        return wrapper
    return decorator

### Here are a whole bunch of type check predicates.

def ALL(*checks):
    def CHK_(x):
        for check in checks:
            check(x)
    return CHK_

def ANY(*checks):
    def CHK_(x):
        for check in checks:
            try:
                check(x)
                return
            except:
                pass
        raise CheckError(x,": failed all checks:",[strc(x) for x in checks])
    return CHK_


@makeargcheck("value should be type book or 0/1")
def BOOL(x):
    return isinstance(x,bool) or (isinstance(x,int) and x in [0,1])

@makeargcheck("value should be an int or a float")
def NUMBER(a):
    return isinstance(a,int) or isinstance(a,float)

def RANGE(lo,hi):
    @makeargcheck("value out of range [%g,%g]"%(lo,hi))
    def RANGE_(x):
        return x>=lo and x<=hi
    return RANGE_

def ARANK(n):
    @makeargcheck("array must have rank %d"%n)
    def ARANK_(a):
        if not hasattr(a,"ndim"): return 0
        return a.ndim==n
    return ARANK_

def ARANGE(lo,hi):
    @makeargcheck("array values must be within [%g,%g]"%(lo,hi))
    def ARANGE_(a):
        return numpy.amin(a)>=lo and numpy.amax(a)<=hi
    return ARANGE_

@makeargcheck("array elements must be non-negative")
def ANONNEG(a):
    return numpy.amin(a)>=0

float_dtypes = [numpy.dtype('float32'),numpy.dtype('float64')]
try: float_dtypes += [numpy.dtype('float96')]
except: pass
try: float_dtypes += [numpy.dtype('float128')]
except: pass

@makeargcheck("array must contain floating point values")
def AFLOAT(a):
    return a.dtype in float_dtypes

int_dtypes = [numpy.dtype('uint8'),numpy.dtype('int32'),numpy.dtype('int64'),numpy.dtype('uint32'),numpy.dtype('uint64')]

@makeargcheck("array must contain integer values")
def AINT(a):
    return a.dtype in int_dtypes

@makeargcheck("expected a byte (uint8) array")
def ABYTE(a):
    return a.dtype==numpy.dtype('B')

@makeargcheck("expect tuple of int")
def inttuple(a):
    if isinstance(a,int): return 1
    if not (tuple(a) or list(a)): return 0
    for x in a:
        if not isinstance(x,int): return 0
    return 1

@makeargcheck("expect tuple of nonnegative int")
def uinttuple(a):
    if isinstance(a,int): return 1
    if not (tuple(a) or list(a)): return 0
    for x in a:
        if not isinstance(x,int): return 0
        if x<0: return 0
    return 1

@makeargcheck("expect pair of int")
def uintpair(a):
    if not tuple(a): return 0
    if not len(a)==2: return 0
    if a[0]<0: return 0
    if a[1]<0: return 0
    return 1

@makeargcheck("expect a rectangle as a pair of slices")
def RECTANGLE(a):
    if not tuple(a): return 0
    if not isinstance(a[0],slice): return 0
    if not isinstance(a[1],slice): return 0
    return 1

### specific kinds of arrays

ARRAY1 = ARANK(1)
ARRAY2 = ARANK(2)
ARRAY3 = ARANK(3)
AINT1 = ALL(ARANK(1),AINT)
AINT2 = ALL(ARANK(2),AINT)
AINT3 = ALL(ARANK(3),AINT)
AFLOAT1 = ALL(ARANK(1),AFLOAT)
AFLOAT2 = ALL(ARANK(2),AFLOAT)
AFLOAT3 = ALL(ARANK(3),AFLOAT)

@makeargcheck("expected a boolean array or an array of 0/1")
def ABINARY(a):
    if a.ndim==2 and a.dtype==numpy.dtype(bool): return 1
    if not a.dtype in int_dtypes: return 0
    import scipy.ndimage.measurements
    zeros,ones = scipy.ndimage.measurements.sum(1,a,[0,1])
    if zeros+ones == a.size: return 1
    if a.dtype==numpy.dtype('B'):
        zeros,ones = scipy.ndimage.measurements.sum(1,a,[0,255])
        if zeros+ones == a.size: return 1
    return 0

ABINARY1 = ALL(ABINARY,ARRAY1)
ABINARY2 = ALL(ABINARY,ARRAY2)
ABINARY3 = ALL(ABINARY,ARRAY3)


def CHANNELS(n):
    @makeargcheck("expected %d channels"%n)
    def CHANNELS_(a):
        return a.shape[-1]==n
    return CHANNELS_

GRAYSCALE = AFLOAT2
GRAYSCALE1 = ALL(AFLOAT2,ARANGE(0,1))
BYTEIMAGE = ALL(ARANK(2),ABYTE)
RGB = ALL(ARANK(3),ABYTE,CHANNELS(3))
RGBA = ALL(ARANK(3),ABYTE,CHANNELS(4))

### image arrays with more complicated image properties

@makeargcheck("expect a light image (median>mean)",warning=1)
def LIGHT(a):
    return numpy.median(a)>=numpy.mean(a)
@makeargcheck("expect a dark image (median<mean)",warning=1)
def DARK(a):
    return numpy.median(a)<=numpy.mean(a)
@makeargcheck("expect a page image (larger than 600x600)",warning=1)
def PAGE(a):
    return a.ndim==2 and a.shape[0]>=600 and a.shape[1]>=600
@makeargcheck("expected a line image (taller than 8 pixels and wider than tall)",warning=1)
def LINE(a,var=None):
    return a.ndim==2 and a.shape[0]>8 # and a.shape[1]>a.shape[0]

BINPAGE = ALL(PAGE,ABINARY2)
LIGHTPAGE = ALL(PAGE,LIGHT)
DARKPAGE = ALL(PAGE,DARK)
LIGHTLINE = ALL(LINE,LIGHT)
DARKLINE = ALL(LINE,DARK)

@makeargcheck("expected a small grayscale patch with values between 0 and 1")
def PATCH(a):
    GRAYSCALE1(a)
    return a.shape[0]<=256 and a.shape[1]<=256

### segmentation-related checks
###
### Segmentations come in two flavors: with a white background (for writing to disk
### so that one can see something in file browsers), and with a black background
### (for easy processing). Light segmentations should only exist on disk.

@makeargcheck("expected a segmentation image")
def SEGMENTATION(a):
    return isinstance(a,numpy.ndarray) and a.ndim==2 and a.dtype in ['int32','int64']
@makeargcheck("expected a segmentation with white background")
def WHITESEG(a):
    return numpy.amax(a)==0xffffff
@makeargcheck("expected a segmentation with black background")
def BLACKSEG(a):
    return numpy.amax(a)<0xffffff
@makeargcheck("all non-zero pixels in a page segmentation must have a column value >0")
def PAGEEXTRA(a):
    u = numpy.unique(a)
    u = u[u!=0]
    u = u[(u&0xff0000)==0]
    return len(u)==0
LIGHTSEG = ALL(SEGMENTATION,WHITESEG)
DARKSEG = ALL(SEGMENTATION,BLACKSEG)
PAGESEG = ALL(SEGMENTATION,BLACKSEG,PAGE,PAGEEXTRA)
LINESEG = ALL(SEGMENTATION,BLACKSEG,LINE)
LIGHTPAGESEG = ALL(SEGMENTATION,WHITESEG,PAGE)
LIGHTLINESEG = ALL(SEGMENTATION,WHITESEG,LINE)

### special types for pattern recognition

def TDATASET(a):
    if not isinstance(a[0], numpy.ndarray):
        raise CheckError("dataset fails to yield ndarray on subscripting")
def DATASET_SIZE(lo=3,hi=int(1e9)):
    @makeargcheck("data set size should be between %s and %s"%(lo,hi))
    def DSSIZE_(a):
        return len(a)>=lo and len(a)<=hi
    return DSSIZE_
def DATASET_VRANK(n):
    @makeargcheck("data set vectors should have a rank of %d"%n)
    def DSVRANK_(a):
        return n<0 or a[0].ndim==n
    return DSVRANK_
def DATASET_VSIZE(lo,hi):
    @makeargcheck("data vector size should be between %d and %d"%(lo,hi))
    def DSVSIZE_(a):
        return a[0].size>=lo and a[0].size<=hi
    return DSVSIZE_
def DATASET_VRANGE(lo,hi):
    @makeargcheck("data set values should be in the range of %g to %g"%(lo,hi))
    def DSVRANGE_(a):
        # just a quick sanity check
        return numpy.amin(a[0])>=lo and numpy.amax(a[0])<=hi
    return DSVRANGE_

def DATASET(size0=3,size1=int(1e9),vsize0=2,vsize1=100000,vrank=-1,vrange0=-300,vrange1=300,fixedshape=0):
    return ALL(TDATASET,
               DATASET_SIZE(size0,size1),
               DATASET_VRANK(vrank),
               DATASET_VSIZE(vsize0,vsize1),
               DATASET_VRANGE(vrange0,vrange1))
