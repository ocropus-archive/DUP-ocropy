import sys,os,re,glob,math,glob,signal,numpy
import iulib,ocropus
from pylab import *

class Record:
    def __init__(self,**kw):
        for k in kw.keys():
            self.__dict__[k] = kw[k]

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

def pad_by(image,r):
    """Symmetrically pad the image by the given amount"""
    w,h = image.shape
    result = zeros((w+2*r,h+2*r))
    result[r:(w+r),r:(h+r)] = image
    return result

def pad_bin(char,r=10):
    """Pad to the next bin size."""
    w,h = char.shape
    w = r*int((w+r-1)/r)
    h = r*int((h+r-1)/r)
    return pad_to(char,w,h)

def number_of_processors():
    try:
        return int(os.popen("cat /proc/cpuinfo  | grep 'processor.*:' | wc -l").read())
    except:
        return 1

def simple_classify(model,inputs):
    result = []
    for i in range(len(inputs)):
        result.append(model.coutputs(inputs[i]))
    return result

def omp_classify(model,inputs):
    if not "ocropus." in str(type(model)):
        return simple_classify(model,inputs)
    omp = ocropus.make_OmpClassifier()
    omp.setClassifier(model)
    n = len(inputs)
    omp.resize(n)
    for i in range(n):
        omp.input(inputs[i],i)
    omp.classify()
    result = []
    for i in range(n):
        outputs = ocropus.OutputVector()
        omp.output(outputs,i)
        outputs = model.outputs2coutputs(outputs)
        result.append(outputs)
    return result

def findfile(name):
    """Find some OCRopus-related resource by looking in a bunch off standard places.
    (FIXME: The implementation is pretty adhoc for now.
    This needs to be integrated better with setup.py and the build system.)"""
    local = "/usr/local/share/ocropus/"
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
    local = "/usr/local/share/ocropus/"
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

def imsdl(image,wait=1,norm=1):
    """Display the given iulib image using iulib's own
    display code."""
    iulib.dinit(512,512,1)
    flag = iulib.dactivate(1)
    if norm:
        iulib.dshown(image)
    else:
        iulib.dshow(image)
    if wait: 
        print "(click to continue)"
        iulib.dwait()
    iulib.dactivate(flag)

def allsplitext(path):
    """Split all the pathname extensions, so that "a/b.c.d" -> "a/b", ".c.d" """
    match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
    if not match:
        return path,""
    else:
        return match.group(1),match.group(3)

