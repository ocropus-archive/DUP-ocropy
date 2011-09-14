### Components from the OCRopus native library that have been replaced
### by the new refactored libraries.

import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology
import common
from common import *

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
        return self.comp.cadd(vector2narray(v),cls)
    def coutputs(self,v,geometry=None):
        """Compute the outputs for a given input vector v.  Outputs are
        of the form [(cls,probability),...]
        The v argument can be rank 1 or larger.  If it is larger, it
        is assumed to be an image and converted from raster to mathematical
        coordinates."""
        # if geometry is not None: warn_once("geometry given to Model")
        return self.comp.coutputs(vector2narray(v))
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
        return self.comp.cclassify(vector2narray(v))

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

class CmodelLineRecognizer(RecognizeLine):
    def __init__(self,**kw):
        """Initialize a line recognizer that works from character models.
        The character shape model is given at initialization and needs to conform to
        the IModel interface.  The segmenter needs to support ISegmentLine.
        The best parameter determines how many of the top outputs from the classifier
        are used in the construction of the lattice.  The maxcost parameter is
        the maximum cost that will be assigned to a transiation in the lattice.
        The reject_cost is a cost above which a character won't get added to the lattice
        at all.  The minheight_letter threshold is the minimum height of a
        component (expressed as fraction of the medium segment height) in
        order to be added as a letter to the lattice."""
        self.cmodel = None
        self.debug = 0
        self.maxspacecost = 20.0
        self.whitespace = "space.model"
        self.segmenter = ocrolseg.DpSegmenter()
        self.grouper = common.StandardGrouper()
        self.nbest = 5
        self.maxcost = 15.0
        self.reject_cost = self.maxcost
        self.min_height = 0.5
        self.rho_scale = 1.0
        self.maxdist = 10
        self.maxrange = 5
        self.use_ligatures = 1
        self.add_rho = 0
        self.verbose = 0
        self.debug_cls = []
        common.set_params(self,kw)
        if type(self.whitespace)==str:
            self.whitespace = common.load_component(common.ocropus_find_file(self.whitespace))
        self.grouper.pset("maxdist",self.maxdist)
        self.grouper.pset("maxrange",self.maxrange)

    def recognizeLine(self,image):
        "Recognize a line, outputting a recognition lattice."""
        lattice,rseg = self.recognizeLineSeg(image)
        return lattice

    def recognizeLineSeg(self,image):
        """Recognize a line.
        lattice: result of recognition
        rseg: intarray where the raw segmentation will be put
        image: line image to be recognized"""

        # first check whether the input dimensions are reasonable

        if image.shape[0]<10:
            raise RecognitionError("line image not high enough (maybe rescale?)",image=image)
        if image.shape[0]>200:
            raise RecognitionError("line image too high (maybe rescale?)",image=image)
        if image.shape[1]<10:
            raise RecognitionError("line image not wide enough (segmentation error?)",image=image)
        if image.shape[1]>10000:
            raise RecognitionError("line image too wide???",image=image)

        # FIXME for some reason, something down below
        # depends on this being a bytearray image, so
        # we're normalizing it here to that type
        image = array(image*255.0/amax(image),'B')

        # compute the raw segmentation
        rseg = self.segmenter.charseg(image)
        if self.debug: show_segmentation(rseg) # FIXME
        rseg = common.renumber_labels(rseg,1) # FIXME
        if amax(rseg)<3: 
            raise RecognitionError("not enough segments in raw segmentation",rseg=rseg)
        self.grouper.setSegmentation(rseg)

        # compute the geometry (might have to use
        # CCS segmenter if this doesn't work well)
        geo = docproc.seg_geometry(rseg)

        # compute the median segment height
        heights = []
        for i in range(self.grouper.length()):
            (y0,x0,y1,x1) = self.grouper.boundingBox(i)
            heights.append(y1-y0)
        mheight = median(array(heights))
        if mheight<8:
            raise RecognitionError("median line height too small (maybe rescale prior to recognition)",mheight=mheight)
        if mheight>100:
            raise RecognitionError("median line height too large (maybe rescale prior to recognition)",mheight=mheight)
        self.mheight = mheight

        # invert the input image (make a copy first)
        old = image
        image = amax(image)-image

        # initialize the whitespace estimator
        self.whitespace.setLine(image,rseg)
        
        # this holds the list of recognized characters if keep!=0
        self.chars = []
        
        # now iterate through the characters
        for i in range(self.grouper.length()):
            # get the bounding box for the character (used later)
            (y0,x0,y1,x1) = self.grouper.boundingBox(i)

            # compute relative geometry
            aspect = (y1-y0)*1.0/(x1-x0)
            try:
                rel = docproc.rel_char_geom((y0,y1,x0,x1),geo)
            except:
                traceback.print_exc()
                raise RecognitionError("bad line geometry",geo=geo)
            ry,rw,rh = rel
            assert rw>0 and rh>0,"error: rw=%g rh=%g"%(rw,rh)
            rel = docproc.rel_geo_normalize(rel)

            # extract the character image (and optionally display it)
            (raw,mask) = self.grouper.extractWithMask(image,i,1)
            char = raw/255.0
            if self.debug:
                imshow(char)
                raw_input()

            # Add a skip transition with the pixel width as cost.
            # This ensures that the lattice is at least connected.
            # Note that for typical character widths, this is going
            # to be much larger than any per-charcter cost.
            if self.add_rho:
                self.grouper.setClass(i,ocropus.L_RHO,self.rho_scale*raw.shape[1])

            # compute the classifier output for this character
            # FIXME parallelize this
            outputs = self.cmodel.coutputs(char,geometry=rel)
            outputs = [(x[0],-log(x[1])) for x in outputs]
            self.chars.append(utils.Record(index=i,image=char,outputs=outputs))

            # estimate the space cost
            sc = self.whitespace.classifySpace(x1)
            yes_space = min(self.maxspacecost,-log(sc[1]))
            no_space = min(self.maxspacecost,-log(sc[0]))

            # maybe add a transition on "_" that we can use to skip 
            # this character if the transcription contains a "~"
            self.grouper.setClass(i,"~",self.reject_cost)
            
            # add the top classes to the lattice
            outputs.sort(key=lambda x:x[1])
            for cls,cost in outputs[:self.nbest]:
                # don't add anything with a cost above maxcost
                # if cost>self.maxcost and cls!="~": continue
                if cls=="~": continue
                if cls in self.debug_cls:
                    print "debug",self.grouper.start(i),self.grouper.end(i),"cls",cls,"cost",cost,\
                        "y %.2f w %.2f h %.2f"%(rel[0],rel[1],rel[2])

                # letters are never small, so we skip small bounding boxes that
                # are categorized as letters; this is an ugly special case, but
                # it is quite common
                category = unicodedata.category(unicode(cls[0]))
                if (y1-y0)<self.min_height*mheight and category[0]=="L":
                    # add an empty transition to allow skipping junk
                    # (commented out right now because I'm not sure whether
                    # the grouper can handle it; FIXME)
                    # self.grouper.setClass(i,"",1.0)
                    continue

                # for anything else, just add the classified character to the grouper
                if self.use_ligatures:
                    ccode = ligatures.lig.ord(cls)
                else:
                    ccode = cls
                self.grouper.setClass(i,ccode,cost)
                self.grouper.setSpaceCost(i,float(yes_space),float(no_space))

        # extract the recognition lattice from the grouper
        lattice = self.grouper.getLattice()

        # return the raw segmentation as a result
        return lattice,rseg

    # align the text line image with the transcription

    def align(self,image,transcription):
        """Takes an image and a transcription and aligns the two.  Output is
        an alignment record, which contains the following fields:
        ins (input symbols), outs (output symbols), costs (corresponding costs)
        cseg (aligned segmentation), rseg (raw segmentation), lattice (recognition lattice),
        raw (raw recognized string without language model), cost (total cost)"""
        lmodel = make_alignment_fst(transcription)
        lattice,rseg = self.recognizeLineSeg(image)
        raw = lattice.bestpath()
        alignment = compute_alignment(lattice,rseg,lmodel)
        alignment.raw = raw
        return alignment

    # saving and loading this component
    
    def save(self,file):
        with open(file,"w") as stream:
            pickle.dump(self,stream)
    def load(self,file):
        with open(file,"r") as stream:
            obj = pickle.load(self,stream)
        self.__dict__.update(obj.__dict__)

    # training is handled separately

    def startTraining(self,type="adaptation"):
        raise Unimplemented()
    def addTrainingLine(self,image,transcription):
        raise Unimplemented()
    def addTrainingLine(self,segmentation,image,transcription):
        raise Unimplemented()
    def finishTraining(self):
        raise Unimplemented()

