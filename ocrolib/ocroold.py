################################################################
### Components from the OCRopus native library that have been replaced
### by the new refactored libraries.
################################################################

import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology
import common
import ocropus
import ocrofst
from iuutils import *
from ocroio import *

################################################################
### native components
################################################################

class ComponentList:
    """Simple interface that lists the native components defined in OCRopus C++ code."""
    def __init__(self):
        self.comp = ocropus.ComponentList()
    def length(self):
        return self.comp.length()
    def kind(self,i):
        return self.comp.kind(i)
    def name(self,i):
        return self.comp.name(i)
    def list(self):
        for i in range(self.length()):
            yield (self.kind(i),self.name(i))

def mknative(spec,interface):
    """Instantiate a native OCRopus component.
    SomeClass:param=value:param=value instantiates a C++ class with no arguments,
    then sets parameters to values using the pset method.
    """
    names = spec.split(":")
    name = names[0]
    exec "constructor = ocropus.make_%s"%interface
    result = constructor(name)
    for param in names[1:]:
        k,v = param.split("=",1)
        try: v = float(v)
        except ValueError: pass
        result.pset(k,v)
    return result

class CommonComponent:
    """Common methods for components.  The implementations in this
    base class is meant for C++ components and wraps those components.
    The methods in the Python classes translate Python datatypes into
    native datatypes and back.  If you implement components in pure
    Python, you do not need to inherit from this."""
    c_interface = None
    c_class = None
    def __init__(self,**kw):
        self.params = kw
        if self.c_class is not None: self.make(self.c_class)
        self.init()
    def init(self):
        pass
    def make(self,comp):
        """Bind this component to an instance of a C++ component."""
        assert self.c_interface is not None
        self.comp = mknative(comp,self.c_interface)
        for k,v in self.params.items(): self.comp.pset(k,v)
        return self
    def load_native(self,file):
        """Load a native C++ component into this Python object."""
        loader = eval("ocropus.load_"+self.c_interface)
        self.comp = loader(file)
        return self
    def save_native(self,file):
        """Save the native C++ component from this object to a file."""
        ocropus.save_component(file,self.comp)
        return self
    def load(self,file):
        self.load_native(file)
    def save(self,file):
        self.save_native(file)
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
        return self.comp.info()
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
    c_interface = "ICleanupGray"
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
    c_interface = "ICleanupBinary"
    def cleanup(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.cleanup(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class RmHalftone(CleanupBinary):
    """Simple algorithm for removing halftones from binary images."""
    c_class = "RmHalftone"

class RmUnderline(CleanupBinary):
    """Simple algorithm for removing underlines from binary images."""
    c_class = "RmUnderLine"

class AutoInvert(CleanupBinary):
    """Simple algorithm for fixing inverted images."""
    c_class = "AutoInvert"

class DeskewPageByRAST(CleanupBinary):
    """Page deskewing for binary images."""
    c_class = "DeskewPageByRAST"

class RmBig(CleanupBinary):
    """Remove connected components that are too big to be text."""
    c_class = "RmBig"

class DocClean(CleanupBinary):
    """Remove document image noise components."""
    c_class = "DocClean"

class PageFrameByRAST(CleanupBinary):
    """Remove elements outside the document page frame."""
    c_class = "PageFrameByRAST"

class Binarize(CommonComponent):
    """Binarize images."""
    c_interface = "IBinarize"
    def binarize(self,page,type='f'):
        """Binarize the image; returns a tuple consisting of the binary image and
        a possibly transformed grayscale image."""
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
    c_class = "StandardPreprocessing"

class BinarizeByRange(Binarize):
    """Simple binarization using the mean of the range."""
    c_class = "BinarizeByRange"

class BinarizeBySauvola(Binarize):
    """Fast variant of Sauvola binarization."""
    c_class = "BinarizeBySauvola"

class BinarizeByOtsu(Binarize):
    """Otsu binarization."""
    c_class = "BinarizeByOtsu"

class BinarizeByHT(Binarize):
    """Binarization by hysteresis thresholding."""
    c_class = "BinarizeByHT"

class TextImageClassification(CommonComponent):
    """Perform text/image classification."""
    c_interface = "ITextImageClassification"
    def textImageProbabilities(self,page):
        result = iulib.intarray()
        self.comp.textImageProbabilities(result,page2narray(page,'B'))
        return narray2pseg(result)

class SegmentPage(CommonComponent):
    """Segment a page into columns and lines (layout analysis)."""
    c_interface = "ISegmentPage"
    def segment(self,page,obstacles=None,black=0):
        page = page2narray(page,'B')
        # iulib.write_image_gray("_seg_in.png",page)
        result = iulib.intarray()
        if obstacles not in [None,[]]:
            raise Unimplemented()
        else:
            self.comp.segment(result,page)
        if black: ocropus.make_page_segmentation_black(result)
        # iulib.write_image_packed("_seg_out.png",result)
        return narray2pseg(result)

class SegmentPageByRAST(SegmentPage):
    """Segment a page into columns and lines using the RAST algorithm."""
    c_class = "SegmentPageByRAST"

class SegmentPageByRAST1(SegmentPage):
    """Segment a page into columns and lines using the RAST algorithm,
    assuming there is only a single column.  This is more robust for
    single column documents than RAST."""
    c_class = "SegmentPageByRAST1"

class SegmentPageBy1CP(SegmentPage):
    """A very simple page segmentation algorithm assuming a single column
    document and performing projection."""
    c_class = "SegmentPageBy1CP"

class SegmentPageByXYCUTS(SegmentPage):
    """An implementation of the XYCUT layout analysis algorithm.  Not
    recommended for production use."""
    c_class = "SegmentPageByXYCUTS"

class SegmentLine(CommonComponent):
    """Segment a line into character parts."""
    c_interface = "ISegmentLine"
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        result = iulib.intarray()
        self.comp.charseg(result,line2narray(line,'B'))
        ocropus.make_line_segmentation_black(result)
        iulib.renumber_labels(result,1)
        return narray2lseg(result)

class DpLineSegmenter(SegmentLine):
    """Segment a text line by dynamic programming."""
    c_class = "DpSegmenter"

class SkelLineSegmenter(SegmentLine):
    """Segment a text line by thinning and segmenting the skeleton."""
    c_class = "SkelSegmenter"

class GCCSLineSegmenter(SegmentLine):
    """Segment a text line by connected components only, then grouping
    vertically related connected components."""
    c_class = "SegmentLineByGCCS"

class CCSLineSegmenter(SegmentLine):
    """Segment a text line by connected components only."""
    c_class = "ConnectedComponentSegmenter"

class RecognizeLine(CommonComponent):
    """A line recognizer in general."""
    c_interface = "IRecognizeLine"
    def recognizeLine(self,line,segmentation=0):
        """Recognizes the line and returns the recognition lattice."""
        fst = OcroFST()
        self.comp.recognizeLine(fst.comp,line2narray(line,'B'))
        return fst
    def recognizeLineSeg(self,line):
        """Recognizes the line and returns the recognition lattice and a segmentation."""
        fst = OcroFST()
        rseg = iulib.intarray()
        self.comp.recognizeLine(rseg,fst.comp,line2narray(line,'B'))
        return (fst,narray2lseg(rseg))
    def startTraining(self,type="adaptation"):
        """Start training the line recognizer."""
        self.comp.startTraining(type)
    def addTrainingLine(self,line,transcription):
        """Add a training line with a transcription."""
        if type(transcription)==list: transcription = "".join(transcription)
        return self.comp.addTrainingLine(line2narray(line),unicode2ustrg(transcription))
    def addTrainingLineSeg(self,segmentation,line,transcription):
        """Add a training line with a transcription and a given segmentation."""
        if type(transcription)==list: transcription = "".join(transcription)
        return self.comp.addTrainingLine(narray2lseg(segmentation),line2narray(line),unicode2ustrg(transcription))
    def finishTraining(self):
        """Finish training (this may go away for a long time as the recognizer is
        being trained."""
        self.comp.finishTraining()
    def align(self,line,transcription):
        """Align the given line image with the transcription.  Returns
        a list of aligned characters, a corresponding segmentation, and the
        corresponding costs."""
        if type(transcription)==list: transcription = "".join(transcription)
        chars = ustrg()
        seg = iulib.intarray()
        costs = iulib.floatarray()
        self.comp.align(chars,seg,costs,line,transcription)
        return Record(outs=ustrg2unicode(chars),cseg=narray2lseg(seg),costs=iulib.numpy(costs,'f'))

class Linerec(RecognizeLine):
    """A line recognizer using neural networks and character size modeling.
    Feature extraction is per character."""
    c_class = "Linerec"

class LinerecExtracted(RecognizeLine):
    """A line recognizer using neural networks and character size modeling.
    Feature extraction is per line."""
    c_class = "LinerecExtracted"

class OmpClassifier:
    """A parallel classifier using OMP.  This only works for native code models.
    You first set the classifier, then you resize to the number of samples you
    want to classify.  Then you set the input vector for each sample and call
    classify().  When classify() returns, you can get the outputs corresponding
    to each input sample."""
    def setClassifier(self,model):
        self.comp = ocropus.make_OmpClassifier()
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
        self.comp.input(common.vector2narray(a),i)
    def output(self,i):
        """Get outputs for input vector i."""
        ov = ocropus.OutputVector()
        self.comp.output(ov,i)
        return self.model.outputs2coutputs(ov)
    def load(self,file):
        """Load a classifier."""
        self.comp = ocropus.OmpClassifier()
        self.comp.load(file)

def ocosts(l):
    """Computes negative log probabilities for coutputs returned by a
    Model and orders by cost."""
    return sorted([(k,-log(v)) for k,v in l],key=lambda x:x[1])

class Model(CommonComponent):
    """A character recognizer in general."""
    c_interface = "IModel"
    def init(self):
        self.omp = None
    def nfeatures(self):
        """Return the expected number of input features."""
        return self.comp.nfeatures()
    def nclasses(self):
        """Return the number of classes."""
        return self.comp.nclasses()
    def setExtractor(self,extractor):
        """Set a feature extractor."""
        self.comp.setExtractor(extractor)
    def updateModel(self,**kw):
        """After adding training samples, train the model."""
        self.comp.updateModel()
    def updateModel1(self,**kw):
        """After adding training samples, train the model. (For Python
        components, this is an iterator that returns intermediate
        results, but here, it just returns.)"""
        self.comp.updateModel()
    def copy(self):
        """Make a copy of this model."""
        c = self.comp.copy()
        m = Model()
        m.comp = c
        return m
    def cadd(self,v,cls,geometry=None):
        """Add a training sample. The cls argument should be a string.
        The v argument can be rank 1 or larger.  If it is larger, it
        is assumed to be an image and converted from raster to mathematical
        coordinates."""
        # if geometry is not None: warn_once("geometry given to Model")
        return self.comp.cadd(common.vector2narray(v),cls)
    def coutputs(self,v,geometry=None):
        """Compute the outputs for a given input vector v.  Outputs are
        of the form [(cls,probability),...]
        The v argument can be rank 1 or larger.  If it is larger, it
        is assumed to be an image and converted from raster to mathematical
        coordinates."""
        # if geometry is not None: warn_once("geometry given to Model")
        return self.comp.coutputs(common.vector2narray(v))
    def coutputs_batch(self,vs,geometries=None):
        if self.omp is None: 
            self.omp = OmpClassifier()
            self.omp.setClassifier(self)
        self.omp.resize(len(vs))
        for i in range(len(vs)):
            self.omp.input(vs[i],i)
        self.omp.classify()
        result = []
        for i in range(len(vs)):
            result.append(self.omp.output(i))
        return result
    def cclassify(self,v,geometry=None):
        """Perform classification of the input vector v."""
        # if geometry is not None: warn_once("geometry given to Model")
        return self.comp.cclassify(common.vector2narray(v))

class OldAutoMlpModel(Model):
    """An MLP classifier trained with gradient descent and
    automatic learning rate adjustment."""
    c_class = "AutoMlpClassifier"

class LatinModel(Model):
    """A classifier that combines several stages: alphabetic classification,
    upper/lower case, ..."""
    c_class = "LatinClassifier"

class Extractor(CommonComponent):
    """Feature extractor. (Only for backwards compatibility; not recommended.)"""
    c_interface = "IExtractor"
    def extract(self,input):
        out = iulib.floatarray()
        self.comp.extract(out,iulib.narray(input,type='f'))
        return iulib.numpy(out,type='f')

class ScaledFE(Extractor):
    c_class = "scaledfe"

class DistComp(CommonComponent):
    """Compute distances fast. (Only for backwards compatibility; not recommended.)"""
    c_interface = "IDistComp"
    def add(self,v):
        self.comp.add(iulib.narray(v,type='f'))
    def distances(self,v):
        out = iulib.floatarray()
        self.comp.add(out,iulib.narray(v,type='f'))
        return iulib.numpy(out,type='f')
    def find(self,v,eps):
        return self.comp.find(iulib.narray(v,type='f'),eps)
    def merge(self,i,v,weight):
        self.comp.merge(i,iulib.narray(v,type='f'),weight)
    def length(self):
        return self.comp.length()
    def counts(self,i):
        return self.comp.counts(i)
    def vector(self,i):
        out = iulib.floatarray()
        self.comp.vector(out,i)
        return iulib.numpy(out,type='f')
    def nearest(self,v):
        return self.comp.find(iulib.narray(v,type='f'))

class EdistComp(DistComp):
    c_class = "edist"

def load_linerec_old(file,wrapper=None):
    """Loads a line recognizer.  This handles a bunch of special cases
    due to the way OCRopus has evolved.  In the long term, .pymodel is the
    preferred format.

    For files ending in .pymodel, just unpickles the contents of the file.

    For files ending in .cmodel, loads the character model using load_IModel
    (it has to be a C++ character classifier), and then instantiates a
    CmodelLineRecognizer with the cmodel as an argument.  Additional parameters
    can be passed as in my.cmodel:best=5.  The line recognizer used can be
    overridden as in my.cmodel:class=MyLineRecognizer:best=17.

    For anything else, uses native load_linerec (which has its own special cases)."""

    if wrapper is None:
        import common
        wrapper = common.CmodelLineRecognizer

    if ".pymodel" in file:
        with open(file,"rb") as stream:
            result = pickle.load(stream)
        if getattr(result,"coutputs",None):
            print "[wrapping %s]"%result
            result = wrapper(cmodel=result)
        print "[loaded %s]"%result
        return result

    if ".cmodel" in file:
        names = file.split(":")
        cmodel = load_IModel(names[0])
        linerec = wrapper(cmodel=cmodel)
        return linerec

    native = ocropus.load_linerec(file)
    assert isinstance(native,ocropus.IRecognizeLine)
    result = RecognizeLine()
    result.comp = native
    return result

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

class Grouper(CommonComponent):
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

# Parallel classification with OMP classifier objects.

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

