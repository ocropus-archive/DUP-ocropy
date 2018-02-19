import inspect
import numpy as np

def summary(x):
    """Summarize a datatype as a string (for display and debugging)."""
    if type(x)==np.ndarray:
        return "<ndarray %s %s>"%(x.shape,x.dtype)
    if type(x)==str and len(x)>10:
        return '"%s..."'%x
    if type(x)==list and len(x)>10:
        return '%s...'%x
    return str(x)


################################################################
### Ocropy exceptions
################################################################

class OcropusException(Exception):
    trace = 1
    def __init__(self,*args,**kw):
        Exception.__init__(self,*args,**kw)

class Unimplemented(OcropusException):
    trace = 1
    "Exception raised when a feature is unimplemented."
    def __init__(self,s):
        Exception.__init__(self,inspect.stack()[1][3])

class Internal(OcropusException):
    trace = 1
    "Exception raised when a feature is unimplemented."
    def __init__(self,s):
        Exception.__init__(self,inspect.stack()[1][3])

class RecognitionError(OcropusException):
    trace = 1
    "Some kind of error during recognition."
    def __init__(self,explanation,**kw):
        self.context = kw
        s = [explanation]
        s += ["%s=%s"%(k,summary(kw[k])) for k in kw]
        message = " ".join(s)
        Exception.__init__(self,message)

class Warning(OcropusException):
    trace = 0
    def __init__(self,*args,**kw):
        OcropusException.__init__(self,*args,**kw)

class BadClassLabel(OcropusException):
    trace = 0
    "Exception for bad class labels in a dataset or input."
    def __init__(self,s):
        Exception.__init__(self,s)

class BadImage(OcropusException):
    trace = 0
    def __init__(self,*args,**kw):
        OcropusException.__init__(self,*args)

class BadInput(OcropusException):
    trace = 0
    def __init__(self,*args,**kw):
        OcropusException.__init__(self,*args,**kw)

class FileNotFound(OcropusException):
    trace = 0
    """Some file-not-found error during OCRopus processing."""
    def __init__(self,fname):
        self.fname = fname
    def __str__(self):
        return "file not found %s"%(self.fname,)
