import sys,os,re,glob,math,glob,signal
import iulib,ocropus,narray
from pylab import *

class Record:
    def __init__(self,**kw):
        for k in kw.keys():
            self.__dict__[k] = kw[k]

def findfile(name):
    """Find some OCRopus-related resource by looking in a bunch off standard places.
    (This needs to be integrated better with setup.py and the build system.)"""
    local = "/usr/local/share/ocropus/"
    path = name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+name
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
        elif skip0 and c==0:
            pass
        elif c<0 or c>=256:
            result += "{%d}"%c
        else:
            result += chr(c)
    return result

### alignment

def rseg_map(inputs):
    """This takes the FST produces by a beam search and looks at
    the input labels.  The input labels contain the correspondence
    between the rseg label and the character.  These are put into
    a dictionary and returned.  This is used for alignment between
    a segmentation and text."""
    n = inputs.length()
    segs = []
    for i in range(n):
        start = inputs.at(i)>>16
        end = inputs.at(i)&0xffff
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

def recognize_and_align(image,linerec,lmodel,beam=10000):
    """Perform line recognition with the given line recognizer and
    language model.  Outputs an object containing the result (as a
    Python string), the costs, the rseg, the cseg, the lattice and the
    total cost.  The recognition lattice needs to have rseg's segment
    numbers as inputs (pairs of 16 bit numbers); SimpleGrouper
    produces such lattices.  cseg==None means that the connected
    component renumbering failed for some reason."""

    ## run the recognizer
    lattice = ocropus.make_OcroFST()
    rseg = iulib.intarray()
    linerec.recognizeLine(rseg,lattice,image)

    ## perform the beam search through the lattice and the model
    v1 = iulib.intarray()
    v2 = iulib.intarray()
    ins = iulib.intarray()
    outs = iulib.intarray()
    costs = iulib.floatarray()
    ocropus.beam_search(v1,v2,ins,outs,costs,lattice,lmodel,beam)

    ## do the conversions
    result = intarray_as_string(outs)

    ## compute the cseg
    rmap = rseg_map(ins)
    cseg = None
    if len(rmap)>1:
        cseg = iulib.intarray()
        cseg.copy(rseg)
        try:
            for i in range(cseg.length()):
                cseg.put1d(i,int(rmap[rseg.at1d(i)]))
        except IndexError:
            raise Exception("renumbering failed")

    ## return everything we computed
    return Record(image=image,
                  output=result,
                  raw=outs,
                  costs=costs,
                  rseg=rseg,
                  cseg=cseg,
                  lattice=lattice,
                  cost=iulib.sum(costs))

def compute_alignment(lattice,rseg,lmodel,beam=10000):
    """Given a lattice produced by a recognizer, a raw segmentation,
    and a language model, computes the best solution, the cseg, and
    the corresponding costs.  These are returned as Python data structures.
    The recognition lattice needs to have rseg's segment numbers as inputs
    (pairs of 16 bit numbers); SimpleGrouper produces such lattices."""

    ## perform the beam search through the lattice and the model
    v1 = narray.intarray()
    v2 = narray.intarray()
    ins = narray.intarray()
    outs = narray.intarray()
    costs = narray.floatarray()
    ocropus.beam_search(v1,v2,ins,outs,costs,lattice,lmodel,beam)

    ## do the conversions
    result = intarray_as_string(outs)

    ## compute the cseg
    rmap = rseg_map(ins)
    cseg = None
    if len(rmap)>1:
        cseg = narray.intarray()
        cseg.copy(rseg)
        try:
            for i in range(cseg.length()):
                cseg.put1d(i,int(rmap[rseg.at1d(i)]))
        except IndexError:
            raise Exception("renumbering failed")

    ## return everything we computed
    return Record(output=result,
                  raw=outs,
                  costs=costs,
                  rseg=rseg,
                  cseg=cseg,
                  lattice=lattice,
                  cost=costs.sum())

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
