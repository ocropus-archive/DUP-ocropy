import numpy
import iulib
import ocropus
import utils
import components
from scipy.misc import imsave
from numpy import *

################################################################
### conversion functions
################################################################

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
        page = a/255.0
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
    checknp(page)
    if v.ndim==1: return iulib.narray(v,type=f)
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
    checknp(page)
    return numpy2narray(line,type=type)

def narray2line(line,type='B'):
    """Convert line narrays to line images."""
    checkna(page)
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
    pseg = pseg[::-1,...]
    pseg = transpose(pseg,[2,1,0])
    return pseg

def lseg2narray(lseg):
    """Convert a page segmentation (rank 3, RGB) to an narray."""
    checknp(lseg)
    assert lseg.dtype=='B' and lseg.ndim==3
    lseg = array(lseg[:,:,0]<<16+lseg[:,:,1]<<8+lseg[:,:,2])
    return numpy2narray(lseg,'i')

def rect2raster(r,h):
    """Convert rectangles to raster coordinates."""
    (x0,y0,x1,y1) = (r.x0,r.y0,r.x1,r.y1)
    y1 = h-y1-1
    y0 = h-y0-1
    return (y1,x0,y0,x1)

def raster2rect(r,h):
    """Convert raster coordinates (row,col,row,col) to rectangles."""
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
### components
################################################################

class CommonComponent:
    """Common methods for components.  The implementations in this
    base class is meant for C++ components and wraps those components.
    The methods in the Python classes translate Python datatypes into
    native datatypes and back.  If you implement components in pure
    Python, you do not need to inherit from this."""
    def __init__(self):
        self.comp = None
    def make(self,name):
        """Bind this component to an instance of a C++ component."""
        self.make_(name)
        return self
    def name(self):
        """Return the name of this component."""
        return self.comp.name()
    def description(self):
        """Return a description of this component."""
        return self.comp.description()
    def interface(self):
        """Return the interface name of this component."""
        return self.comp.interface()
    def print_(self):
        """Print a representation of this object."""
        return self.comp.print_()
    def info(self,depth=0):
        """Print information about this object."""
        return self.comp.info(depth=0)
    def save_native(self,file):
        """Save this component to FILE as a C++ saved object."""
        raise Exception("unimplemented")
    def load_native(self,file):
        """Load this component from FILE. This may change
        the underlying native component."""
        raise Exception("unimplemented")
    def pexists(self,name):
        """Check whether parameter NAME exists."""
        return self.comp.pexists(name)
    def pset(self,name,value):
        """Set parameter NAME to VALUE."""
        return self.comp.pset(name,value)
    def pget(self,name):
        """Get the value of string parameter NAME."""
        return self.comp.pget(name)
    def pgetf(self,name):
        """Get the value of floating point parameter NAME."""
        return self.comp.pgetf(name)
    def command(self,*args):
        """Send the arguments as a command to the object and return the result."""
        return self.comp.command(*args)
    def plength(self):
        """Get the number of parameters."""
        return self.comp.plength()
    def pname(self,i):
        """Get the name of parameter I."""
        return self.comp.pname(i)
    def reinit(self):
        """Reinitialize the C++ component (if supported)."""
        self.comp.reinit()

class CleanupGray(CommonComponent):
    """Cleanup grayscale images."""
    def make_(self,name):
        self.comp = components.make_ICleanupGray(name,dtype='B')
    def cleanup_gray(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.cleanup_gray(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class DeskewGrayPageByRAST(CleanupGray):
    """Page deskewing for gray scale images."""
    def __init__(self):
        self.make_("DeskewGrayPageByRAST")

class CleanupBinary(CommonComponent): 
    """Cleanup binary images."""
    def make_(self,name):
        self.comp = components.make_ICleanupBinary(name)
    def cleanup(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.cleanup(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class RmHalftone(CleanupBinary):
    """Simple algorithm for removing halftones from binary images."""
    def __init__(self):
        self.make_("RmHalftone")
class RmUnderline(CleanupBinary):
    """Simple algorithm for removing underlines from binary images."""
    def __init__(self):
        self.make_("RmUnderline")
class AutoInvert(CleanupBinary):
    """Simple algorithm for fixing inverted images."""
    def __init__(self):
        self.make_("AutoInvert")
class DeskewPageByRAST(CleanupBinary):
    """Page deskewing for binary images."""
    def __init__(self):
        self.make_("DeskewPageByRAST")
class RmBig(CleanupBinary):
    """Remove connected components that are too big to be text."""
    def __init__(self):
        self.make_("RmBig")
class DocClean(CleanupBinary):
    """Remove document image noise components."""
    def __init__(self):
        self.make_("DocClean")
class PageFrameByRAST(CleanupBinary):
    """Remove elements outside the document page frame."""
    def __init__(self):
        self.make_("PageFrameByRAST")

class Binarize(CommonComponent):
    """Binarize images."""
    def make_(self,name):
        self.comp = components.make_IBinarize(name)
    def binarize(self,page,type='f'):
        if len(page.shape)==3: page = mean(page,axis=2)
        bin = iulib.bytearray()
        gray = iulib.bytearray()
        self.comp.binarize(bin,gray,page2narray(page,'B'))
        return (narray2page(bin,type=type),narray2page(gray,type=type))
    def binarize_color(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.binarize_color(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class StandardPreprocessing(Binarize):
    """Complete pipeline of deskewing, binarization, and page cleanup."""
    def __init__(self):
        self.make_("StandardPreprocessing")
class BinarizeByRange(Binarize):
    """Simple binarization using the mean of the range."""
    def __init__(self):
        self.make_("BinarizeByRange")
class BinarizeBySauvola(Binarize):
    """Fast variant of Sauvola binarization."""
    def __init__(self):
        self.make_("BinarizeBySauvola")
class BinarizeByOtsu(Binarize):
    """Otsu binarization."""
    def __init__(self):
        self.make_("BinarizeByOtsu")
class BinarizeByHT(Binarize):
    """Binarization by hysteresis thresholding."""
    def __init__(self):
        self.make_("BinarizeByHT")

class TextImageClassification(CommonComponent):
    """Perform text/image classification."""
    def make_(self,name):
        self.comp = components.make_ICleanupBinary(name)
    def textImageProbabilities(self,page):
        result = iulib.intarray()
        self.comp.textImageProbabilities(result,page2narray(page,'B'))
        return narray2pseg(result)

class SegmentPage(CommonComponent):
    """Segment a page into columns and lines (layout analysis)."""
    def make_(self,name):
        self.comp = components.make_ISegmentPage(name)
    def segment(self,page,obstacles=None):
        page = page2narray(page,'B')
        iulib.write_image_gray("_seg_in.png",page)
        result = iulib.intarray()
        if obstacles not in [None,[]]:
            raise Exception("unimplemented")
        else:
            self.comp.segment(result,page)
        # ocropus.make_page_segmentation_black(result)
        iulib.write_image_packed("_seg_out.png",result)
        return narray2pseg(result)

class SegmentPageByRAST(SegmentPage):
    """Segment a page into columns and lines using the RAST algorithm."""
    def __init__(self):
        self.make_("SegmentPageByRAST")
class SegmentPageByRAST1(SegmentPage):
    """Segment a page into columns and lines using the RAST algorithm,
    assuming there is only a single column.  This is more robust for
    single column documents than RAST."""
    def __init__(self):
        self.make_("SegmentPageByRAST1")
class SegmentPageBy1CP(SegmentPage):
    """A very simple page segmentation algorithm assuming a single column
    document and performing projection."""
    def __init__(self):
        self.make_("SegmentPageBy1CP")
class SegmentPageByXYCUTS(SegmentPage):
    """An implementation of the XYCUT layout analysis algorithm.  Not
    recommended for production use."""
    def __init__(self):
        self.make_("SegmentPageByXYCUTS")

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
        iulib.write_image_packed("_seg.png",image)
        self.comp.setPageLines(image)
    def id(self,i):
        """Return the RGB pixel value for this segment."""
        return self.comp.id(i)
    def x0(self):
        """Return x0 (column) for the start of the box."""
        return self.comp.x0(i)
    def x1(self):
        """Return x0 (column) for the end of the box."""
        return self.comp.x1(i)
    def y0(self):
        """Return y0 (row) for the start of the box."""
        return h-self.comp.y1(i)-1
    def y1(self):
        """Return y0 (row) for the end of the box."""
        return h-self.comp.y0(i)-1
    def bbox(self,i):
        """Return the bounding box in raster coordinates
        (row0,col0,row1,col1)."""
        r = self.comp.bbox(i)
        return rect2raster(r,self.h)
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
    def extractMasked(self,image,index,grow,bg,margin=0,type=None):
        """Return the masked subimage for component index, elsewhere the bg value."""
        h,w = image.shape[:2]
        (r0,c0,r1,c1) = self.bbox(index)
        mask = self.mask(index,margin=margin)
        subimage = image[max(0,r0-margin):min(h,r1+margin),max(0,c0-margin):min(w,c1+margin),...]
        return where(mask,subimage,bg)

class SegmentLine(CommonComponent):
    """Segment a line into character parts."""
    def make_(self,name):
        self.comp = components.make_ISegmentLine(name)
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        result = iulib.intarray()
        self.comp.segment(result,line2narray(line,'B'))
        return narray2lseg(result)

class DpLineSegmenter(SegmentLine):
    """Segment a text line by dynamic programming."""
    def __init__(self):
        self.make_("DpSegmenter")
class SkelLineSegmenter(SegmentLine):
    """Segment a text line by thinning and segmenting the skeleton."""
    def __init__(self):
        self.make_("SkelSegmenter")
class GCCSLineSegmenter(SegmentLine):
    """Segment a text line by connected components only, then grouping
    vertically related connected components."""
    def __init__(self):
        self.make_("SegmentLineByGCCS")
class CCSLineSegmenter(SegmentLine):
    """Segment a text line by connected components only."""
    def __init__(self):
        self.make_("ConnectedComponentSegmenter")

class Grouper(CommonComponent):
    """Perform grouping operations on segmented text lines, and
    create a finite state transducer for classification results."""
    def make_(self,name):
        self.comp = components.make_IGrouper(name)
    def setSegmentation(self,segmentation):
        """Set the line segmentation."""
        self.comp.setSegmentation(line2narray(segmentation,'B'))
        self.h = segmentation.shape[0]
    def setCSegmentation(self,segmentation):
        """Set the line segmentation, assumed to be a cseg."""
        self.comp.setCSegmentation(line2narray(segmentation,'B'))
        self.h = segmentation.shape[0]
    def length(self):
        """Number of groups."""
        return self.comp.length()
    def getMask(self,i,margin=0):
        """Get the mask image for group i."""
        rect = rectangle()
        mask = iulib.bytearray()
        self.comp.getMask(rect,mask,i,margin)
        return (rect2raster(rect,self.h),narray2numpy(mask,'f'))
    def getMaskAt(self,i,rect):
        """Get the mask for group i and contained in the given rectangle."""
        rect = raster2rect(rect,self.h)
        mask = iulib.bytearray()
        self.comp.getMaskAt(mask,i,rect)
        return narray2numpy(mask,'f')
    def boundingBox(self,i):
        """Get the bounding box for group i."""
        return rect2raster(self.comp.boundingBox(i),self.h)
    def start(self,i):
        """Get the identifier of the character segment starting this group."""
        return self.start(i)
    def end(self,i):
        """Get the identifier of the character segment ending this group."""
        return self.end(i)
    def getSegments(self,i):
        """Get a list of all the segments making up this group."""
        l = iulib.intarray()
        return [l.at(i) for i in range(l.length())]
    def extract(self,source,dflt,i,grow=0,dtype='f'):
        """Extract the image corresponding to group i.  Background pixels are
        filled in with dflt."""
        if isfp(source):
            out = iulib.floatarray()
            self.comp.extract(out,numpy2narray(source,'f'),dflt,i,grow)
            return narray2numpy(out,'f')
        else:
            out = iulib.bytearray()
            self.comp.extract(out,numpy2narray(source,'f'),dflt,i,grow)
            return narray2numpy(out,'B')
    def extractWithMask(self,source,i,grow=0):
        """Extract the image and mask corresponding to group i"""
        if isfp(source):
            out = iulib.floatarray()
            mask = iulib.bytearray()
            self.comp.extractWithMask(out,mask,numpy2narray(source,'f'),i,grow)
            return (narray2numpy(out,'f'),narray2numpy(out,'b'))
        else:
            out = iulib.bytearray()
            mask = iulib.bytearray()
            self.comp.extractWithMask(out,mask,numpy2narray(source,'B'),i,grow)
            return (narray2numpy(out,'b'),narray2numpy(out,'B'))
    def extractSliced(self,source,dflt,i,grow=0):
        """Extract the image and mask corresponding to group i, slicing through the entire input
        line.  Background pixels are filled with dflt."""
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
        if type(cls)==str:
            self.comp.setClass(i,unicode2ustrg(unicode(cls)),cost)
        elif type(cls)==unicode:
            self.comp.setClass(i,unicode2ustrg(cls),cost)
        elif type(cls)==int:
            self.comp.setClass(i,cls,cost)
        else:
            raise Exception("bad class type")
    def setSpaceCost(self,i,yes_cost,no_cost):
        """Set the cost of putting a space or not putting a space after
        group i."""
        self.comp.setSpaceCost(i,yes_cost,no_cost)
    def getLattice(self,fst):
        """Construct the lattice for the group, using the setClass and setSpaceCost information."""
        self.comp.getLattice(fst)
    def clearLattice(self):
        """Clear all the lattice-related information accumulated so far."""
        self.comp.clearLattice()
    def pixelSpace(self,i):
        """???"""
        return self.comp.pixelSpace(i)

class StandardGrouper(Grouper):
    """The grouper usually used for printed OCR."""
    def __init__(self):
        self.make_("StandardGrouper")

class RecognizeLine(CommonComponent):
    """A line recognizer in general."""
    def make_(self,name):
        self.comp = components.make_IRecognizeLine(name)
    def recognizeLine(self,fst,line,segmentation=0):
        """Recognizes the line, saving the lattice into fst, and returning a segmentation."""
        if segmentation:
            segmentation = iulib.intarray()
            self.comp.recognizeLine(fst,segmentation,line2narray(line,'B'))
            return narray2lseg(segmentation)
        else:
            self.comp.recognizeLine(fst,line2narray(line,'B'))
    def startTraining(self,type="adaptation"):
        """Start training the line recognizer."""
        self.comp.startTraining(type)
    def addTrainingLine(self,line,transcription):
        """Add a training line with a transcription."""
        self.comp.addTrainingLine(line2narray(line),unicode2ustrg(transcription))
    def addTrainingLine(self,segmentation,line,transcription):
        """Add a training line with a transcription and a given segmentation."""
        raise Exception("unimplemented")
    def finishTraining(self):
        """Finish training (this may go away for a long time as the recognizer is
        being trained."""
        self.comp.finishTraining()
    def align(self,line,transcription):
        """Align the given line image with the transcription.  Returns
        a list of aligned characters, a corresponding segmentation, and the
        corresponding costs."""
        chars = ustrg()
        seg = iulib.intarray()
        costs = iulib.floatarray()
        self.comp.align(chars,seg,costs,line,transcription)
        return (ustrg2unicode(chars),narray2lseg(seg),iulib.numpy(costs,'f'))

class Linerec(RecognizeLine):
    """A line recognizer using neural networks and character size modeling.
    Feature extraction is per character."""
    def __init__(self):
        self.make_("Linerec")
class LinerecExtracted(RecognizeLine):
    """A line recognizer using neural networks and character size modeling.
    Feature extraction is per line."""
    def __init__(self):
        self.make_("linerec_extracted")

class Model(CommonComponent):
    """A classifier in general."""
    def make_(self,name):
        self.comp = components.make_IModel(name)
    def nfeatures(self):
        """Return the expected number of input features."""
        return self.comp.nfeatures()
    def nclasses(self):
        """Return the number of classes."""
        return self.comp.nclasses()
    def setExtractor(self,extractor):
        """Set a feature extractor."""
        self.comp.setExtractor(extractor)
    def updateModel(self):
        """After adding training samples, train the model."""
        self.comp.updateModel()
    def copy(self):
        """Make a copy of this model."""
        c = self.comp.copy()
        m = Model()
        m.comp = c
        return m
    def cadd(self,v,cls):
        """Add a training sample. The cls argument should be a string.
        The v argument can be rank 1 or larger.  If it is larger, it
        is assumed to be an image and converted from raster to mathematical
        coordinates."""
        return self.comp.cadd(vector2narray(v),cls)
    def coutputs(self,v):
        """Compute the ouputs for a given input vector v.  Outputs are
        of the form [(cls,probability),...]
        The v argument can be rank 1 or larger.  If it is larger, it
        is assumed to be an image and converted from raster to mathematical
        coordinates."""
        return self.comp.coutputs(vector2narray(v),coutputs)
    def cclassify(self,v):
        """Perform classification of the input vector v."""
        return self.comp.cclassify(vector2narray(v))

class KnnClassifier(CommonComponent):
    """A simple nearest neighbor classifier."""
    def __init__(self):
        self.make_("KnnClassifier")
class AutoMlpClassifier(CommonComponent):
    """An MLP classifier trained with gradient descent and
    automatic learning rate adjustment."""
    def __init__(self):
        self.make_("AutoMlpClassifier")
class LatinClassifier(CommonComponent):
    """A classifier that combines several stages: alphabetic classification,
    upper/lower case, ..."""
    def __init__(self):
        self.make_("LatinClassifier")

class OmpClassifier:
    """A parallel classifier using OMP.  This only works for native code models.
    You first set the classifier, then you resize to the number of samples you
    want to classify.  Then you set the input vector for each sample and call
    classify().  When classify() returns, you can get the outputs corresponding
    to each input sample."""
    def setClassifier(self,model):
        self.comp = ocropus.OmpClassifier()
        if isinstance(model,Model):
            self.model = model.comp
            self.comp.setClassifier(model.comp)
        else:
            self.model = model
            self.comp.setClassifier(model)
    def resize(self,i):
        """Make room for i input vectors."""
        self.comp.resize(i)
    def classify(self):
        """Perform classification in parallel for all the input
        vectors that have been set."""
        self.comp.classify()
    def input(self,a,i):
        """Set input vector i."""
        self.comp.input(vector2narray(a),i)
    def output(self,i):
        """Get outputs for input vector i."""
        ov = ocropus.OutputVector()
        self.comp.output(ov,i)
        return self.model.outputs2coutputs(ov)
    def load(self,file):
        """Load a classifier."""
        self.comp = ocropus.OmpClassifier()
        self.comp.load(file)

class OcroFST():
    def __init__(self):
        self.comp = ocropus.make_OcroFST()
    def clear(self):
        self.comp.clear()
    def newState(self):
        return self.comp.newState()
    def addTransition(frm,to,output,cost=0.0,inpt=None):
        if inpt is None:
            self.comp.addTransition(frm,to,output,cost)
        else:
            self.comp.addTransition(frm,to,output,cost,inpt)
    def setStart(self,node):
        self.comp.setStart(node)
    def setAccept(self,node,cost=0.0):
        self.comp.setAccept(node,cost)
    def special(self,s):
        return self.comp.special(s)
    def bestpath(self):
        result = iulib.ustrg()
        self.comp.bestpath(result)
        ustrg2unicode(result)
    def setString(self,s,costs,ids):
        self.comp.setString(unicode2ustrg(s),costs,ids)
    def nStates(self):
        return self.comp.nStates()
    def getStart(self):
        return self.comp.getStart()
    def getAcceptCost(self,node):
        return self.comp.getAcceptCost(node)
    def isAccepting(self,node):
        return self.comp.isAccepting(node)
    def getTransitions(self,frm):
        tos = iulib.intarray()
        symbols = iulib.intarray()
        costs = iulib.floatarray()
        inputs = iulib.intarray()
        self.comp.getTransitions(tos,symbols,costs,inputs,frm)
        return (iulib.numpy(tos,'i'),iulib.numpy(symbols,'i'),iulib.numpy(costs),iulib.numpy(inputs,'i'))
    def rescore(frm,to,symbol,new_cost,inpt=None):
        if inpt is None:
            self.comp.resocre(frm,to,symbol,new_cost)
        else:
            self.comp.resocre(frm,to,symbol,new_cost,inpt)
    def load(self,name):
        self.comp.load(name)
    def save(self,name):
        self.comp.save(name)
    def as_openfst(self):
        import openfst,os
        tmpfile = "/tmp/%d.fst"%os.getpid()
        self.comp.save(tmpfile)
        fst = openfst.StdVectorFst()
        fst.Read(tmpfile)
        os.unlink(tmpfile)
        return fst

def page_iterator(files):
    """Given a list of image files, iterate through all images in that list.
    In particular, if any files are multipage TIFF images, load each page
    contained in that TIFF image."""
    # use the iulib implementation because its TIFF reader actually works;
    # the TIFF reader in all the Python libraries is broken
    for image,file in utils.page_iterator(files):
        yield narray2page(image),file

def read_image_gray(file,type='B'):
    """Read an image in grayscale."""
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
    
def read_page_segmentation(name,pseg,black=1):
    """Write a numpy page segmentation (rank 3, type='B' RGB image.)"""
    pseg = iulib.intarray()
    iulib.read_image_packed(pseg,name)
    if black: ocropus.make_page_segmentation_black(pseg)
    return narray2pseg(pseg)
    
def write_line_segmentation(name,pseg,white=1):
    """Write a numpy line segmentation."""
    pseg = pseg2narray(pseg)
    if white: ocropus.make_line_segmentation_white(pseg)
    iulib.write_image_packed(name,pseg)
    
def read_line_segmentation(name,pseg,black=1):
    """Write a numpy line segmentation."""
    pseg = iulib.intarray()
    iulib.read_image_packed(pseg,name)
    if black: ocropus.make_line_segmentation_black(pseg)
    return narray2pseg(pseg)
    
