import sys,os,re,glob,math,glob,signal
import iulib,ocropus,narray
from pylab import *

class Record:
    def __init__(self,**kw):
        for k in kw.keys():
            self.__dict__[k] = kw[k]

def number_of_processors():
    try:
        return int(os.popen("cat /proc/cpuinfo  | grep 'processor.*:' | wc -l").read())
    except:
        return 1

def omp_classify(model,inputs):
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

def N(image):
    """Convert an narray to a numpy array."""
    return iulib.numpy(image)

def NI(image):
    """Convert an narray to a numpy array, accounting for
    different coordinate conventions."""
    return transpose(N(image))[::-1,...]

def F(image):
    """Convert a numpy array to an iulib floatarray."""
    a = iulib.floatarray()
    iulib.narray_of_numpy(a,image)
    return a

def FI(image):
    """Convert a numpy array to an iulib floatarray, accounting
    for different coordinate conventions."""
    a = iulib.floatarray()
    iulib.narray_of_numpy(a,transpose(image[::-1,...]))
    return a

def ustrg_as_string(s,skip0=1):
    ## FIXME: handle unicode
    """Convert an iulib ustrg into a Python string"""
    result = ""
    for i in range(s.length()):
        c = s.ord(i)
        if c==ocropus.L_RHO:
            result += "~"
        elif skip0 and c==0:
            pass
        elif c<0 or c>=256:
            result += "{%d}"%c
        else:
            result += chr(c)
    return result

def intarray_as_string(s,skip0=1):
    ## FIXME: handle unicode
    """Convert an iulib intarray into a Python string"""
    result = ""
    for i in range(s.length()):
        c = s.at(i)
        if c==ocropus.L_RHO:
            result += "~"
        elif c==0:
            if skip0:
                pass
            else:
                result += "_"
        elif c<0 or c>=256:
            result += "{%d}"%c
        else:
            result += chr(c)
    return result

def write_line_segmentation(file,seg_):
    """Write the line segmentation to the output file, changing black
    background to write."""
    seg = iulib.intarray()
    seg.copy(seg_)
    ocropus.make_line_segmentation_white(seg)
    iulib.write_image_packed(file,seg)

def write_text(file,s):
    """Write the given string s to the output file."""
    with open(file,"w") as stream:
        stream.write(s)

def allsplitext(path):
    """Split all the pathname extensions, so that "a/b.c.d" -> "a/b", ".c.d" """
    match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
    if not match:
        return path,""
    else:
        return match.group(1),match.group(3)

from PIL import Image

def page_iterator(files):
    for file in files:
        _,ext = os.path.splitext(file)
        if ext.lower()==".tif" or ext.lower()==".tiff":
            if os.path.getsize(file)>2e9:
                raise IOError("TIFF file is greater than 2G")
            if 1:
                tiff = iulib.Tiff(file,"r")
                for i in range(tiff.numPages()):
                    image = iulib.bytearray()
                    try:
                        tiff.getPageRaw(image,i,True)
                    except:
                        tiff.getPage(image,i,True)
                    yield image,"%s[%d]"%(file,i)
            else:
                image = Image.open(file)
                for i in range(10000):
                    try:
                        image.seek(i)
                    except EOFError:
                        break
                    sys.stderr("# TIFF frame %d\n"%image.tell())
                    frame = array(image)
                    frame = ocropy.FI(frame)
                    yield frame,"%s[%d]"%(file,i)
        else:
            image = iulib.bytearray()
            iulib.read_image_gray(image,file)
            yield image,file

def show_segmentation(rseg):
    """Shows a line or page segmentation using Matplotlib's imshow.
    The argument should be an narray."""
    temp = iulib.numpy(rseg,type='B')
    temp[temp==255] = 0
    temp = transpose(temp)[::-1,:]
    temp2 = 1 + (temp % 10)
    temp2[temp==0] = 0
    temp = temp2
    print temp.shape,temp.dtype
    temp = temp/float(amax(temp))
    imshow(temp,cmap=cm.spectral)

            
