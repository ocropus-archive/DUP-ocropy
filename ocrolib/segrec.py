################################################################
### Segmenting line recognizer.  This is the main recognizer
### in OCRopus right now.
################################################################

import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from pylab import randn
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology

import docproc
import ligatures
import ocrorast
import ocrolseg
import ocropreproc
import common
import grouper
from pycomp import PyComponent
from ocroio import renumber_labels
from pylab import *

import cPickle as pickle
pickle_mode = 2



class ClassifierModel(PyComponent):
    """Wraps all the necessary functionality around a classifier in order to
    turn it into a character recognition model."""
    def __init__(self,**kw):
        self.nbest = 5
        self.minp = 1e-3
        self.classifier = self.makeClassifier()
        self.extractor = self.makeExtractor()
        kw = common.set_params(self,kw)
        kw = common.set_params(self.classifier,kw)
        self.rows = None
        self.nrows = 0
        self.classes = []
        self.c2i = {}
        self.i2c = []
        self.geo = None

    def set(self,**kw):
        kw = common.set_params(self,kw)
        kw = common.set_params(self.classifier,kw)
        assert kw=={},"extra parameters to %s: %s"%(self,kw)

    ## helper methods

    def setupGeometry(self,geometry):
        """Set the self.geo instance variable to remember whether this
        Model uses geometry or not."""
        if self.geo is None:
            if geometry is None: 
                self.geo = 0
            else:
                self.geo = len(array(geometry,'f'))
    def makeInput(self,image,geometry):
        """Given an image and geometry, compute and return a feature vector.
        This calls the feature extractor via self.extract(image)."""
        v = self.extractor.extract(image).ravel()
        if self.geo>0:
            if geometry is not None:
                geometry = array(geometry,'f')
                assert len(geometry)==self.geo
                v = concatenate([v,geometry])
        return v
    def makeOutputs(self,w):
        """Given an output vector w from a classifier, translate use the
        i2c array to translate this into a list [(class,probability),...]
        representing the string output of the character recognizer."""
        result = []
        indexes = argsort(-w)
        for i in indexes[:self.nbest]:
            if w[i]<self.minp: break
            result.append((self.i2c[i],w[i]))
        return result

    ## public methods
    
    def clear(self):
        """Completely clear the classifier"""
        self.rows = None
        self.classes = None
        self.nrows = 0

    def cadd(self,image,c,geometry=None):
        """Add a character to the model for training.  The image may be of variable size.
        c should be the corresponding class, a string.  If geometry is given, it must be
        given always and consistently."""
        common.check_valid_class_label(c)
        if self.geo is None: 
            # first time around, remember whether this classifier uses geometry
            self.setupGeometry(geometry)
        v = self.makeInput(image,geometry)
        assert amin(v)>=-1.2 and amax(v)<=1.2
        if self.nrows==0:
            self.rows = zeros((1000,len(v)),'int8')
        elif self.nrows==len(self.rows):
            n,d = self.rows.shape
            self.rows.resize((1000+n,d))
        self.rows[self.nrows,:] = 100.0*v
        if c not in self.c2i:
            self.c2i[c] = len(self.i2c)
            self.i2c.append(c)
        self.classes.append(self.c2i[c])
        self.nrows += 1

    def updateModel(self,*args,**kw):
        """Perform actual training of the model."""
        n,d = self.rows.shape
        self.rows.resize(self.nrows,d)
        self.classifier.train(self.rows,array(self.classes,'i'),*args,**kw)
        self.clear()

    def updateModel1(self,*args,**kw):
        """Perform training of the model.  This actually is an iterator and
        returns from time to time during training to allow saving of intermediate
        models.  Use as in "for progress in model.updateModel1(): print progress" """
        if not hasattr(self.classifier,"train1"):
            warn_once("no train1 method; just doing training in one step")
            self.updateModel(*args,**kw)
            return
        n,d = self.rows.shape
        self.rows.resize(self.nrows,d)
        for progress in self.classifier.train1(self.rows,array(self.classes,'i'),*args,**kw):
            yield progress
        self.clear()

    def coutputs(self,image,geometry=None):
        """Given an image and corresponding geometry (as during
        training), compute outputs (posterior probabilities or
        discriminant functions) for the image.  Returns a list like
        [(class,probability),...]"""
        assert (not self.geo) or (geometry is not None),\
            "classifier requires geometry but none is given"
        v = self.makeInput(image,geometry)
        w = self.classifier.outputs(v.reshape(1,len(v)))[0]
        return self.makeOutputs(w)

    def cclassify(self,v,geometry=None):
        """Given an image and corresponding geometry (as during
        training), classify the image.  Returns just the class."""
        assert (not self.geo) or (geometry is not None),\
            "classifier requires geometry but none is given"
        v = self.makeInput(image,geometry)
        w = self.classifier.outputs(v.reshape(1,len(v)))[0]
        return self.i2c[argmax(w)]

    def coutputs_batch(self,images,geometries=None):
        """Given a list of images (and an optional list of geometries if the
        classifier requires it), compute a list of the corresponding outputs.
        This is the same as calling coutputs repeatedly, but it may be
        parallelized."""
        # FIXME parallelize this
        if geometries is None: geometries = [None]*len(images)
        result = []
        for i in range(len(images)):
            try:
                output = self.coutputs(images[i],geometries[i]) 
            except:
                print "recognition failed"
                output = []
            result.append(output)
        return result

    def save_component(self,path):
        """Save this component to a file (using pickle). Use
        ocrolib.load_component or cPickle.load to read the component
        back in again."""
        rows = self.rows
        nrows = self.nrows
        classes = self.classes
        self.rows = None
        self.classes = None
        self.nrows = None
        with open(path,"wb") as stream:
            pickle.dump(self,stream,pickle_mode)
        self.rows = rows
        self.classes = classes
        self.nrows = nrows

class BboxFE(PyComponent):
    """A feature extractor that only rescales the input image to fit into
    a 32x32 (or, generally, r x r box) and normalizes the vector.
    Parameters are r (size of the rescaled image), and normalize (can be
    one of "euclidean", "max", "sum", or None)."""
    def __init__(self,**kw):
        self.r = 32
        self.normalize = None
        common.set_params(self,kw)
    def extract(self,image):
        v = array(docproc.isotropic_rescale(image,self.r),'f')
        if not hasattr(self,"normalize") or self.normalize is None:
            pass
        elif self.normalize=="euclidean":
            v /= sqrt(sum(v**2))
        elif self.normalize=="max":
            v /= amax(v)
        elif self.normalize=="sum":
            v /= sum(abs(v))
        return v

class Classifier(PyComponent):
    """An abstraction for a classifier.  This gets trained on training vectors and
    returns vectors of posterior probabilities (or some other discriminant function.)
    You usually save these objects by pickling them."""
    def train(self,data,classes):
        """Train the classifier on the given dataset."""
        raise Unimplemented()
    def outputs(self,data):
        """Compute the ouputs corresponding to each input data vector."""
        raise Unimplemented()

class SegWithCost:
    def __init__(self):
        self.segmenter0 = ocrolseg.SegmentLineByGCCS()
        self.segmenter1 = ocrolseg.DpSegmenter()
    def segment(self,image):
        seg0 = self.segmenter0.charseg(image)
        assert amax(seg0)<32000
        seg1 = self.segmenter1.charseg(image)
        assert amax(seg1)<32000
        cor = ((seg0<<16)|seg1)
        hist = Counter(cor.ravel())
        

class CmodelLineRecognizer:
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
        self.norejects = 0
        self.gccs = 0
        self.cmodel = None
        self.display = 0
        self.display_shape = (7,7)
        self.minsegs = 3
        self.spaces = 1 # add spaces (turn off for debugging)
        self.maxspacecost = 20.0
        self.whitespace = "space.model"
        self.nbest = 5 # use at most this many outputs
        self.maxcost = 15.0
        self.reject_cost = self.maxcost
        self.min_height = 0.5
        self.rho_scale = 1.0
        self.maxdist = 2
        self.use_ligatures = 0
        self.add_rho = 0
        self.verbose = 0
        self.debug_cls = None
        self.allow_any = 0 # allow non-unicode characters
        self.combined_cost = 2.0 # extra cost for combining connected components
        self.split_cost = 0.0 # extra cost for combining connected components
        self.maxrange = 4
        self.segmenter = ocrolseg.DpSegmenter()
        self.segmenter0 = ocrolseg.SegmentLineByGCCS()
        common.set_params(self,kw)
        if type(self.whitespace)==str:
            self.whitespace = common.load_component(common.ocropus_find_file(self.whitespace))
        self.grouper = grouper.Grouper()
        self.grouper.pset("maxdist",self.maxdist)
        self.grouper.pset("maxrange",self.maxrange)

    def recognize(self,image):
        """Recognize a line. Leaves the results in self.grouper and self.rseg."""

        # first check whether the input dimensions are reasonable

        if image.shape[0]<10:
            raise common.RecognitionError("line image not high enough (maybe rescale?)",image=image)
        if image.shape[0]>200:
            raise common.RecognitionError("line image too high (maybe rescale?)",image=image)
        if image.shape[1]<10:
            raise common.RecognitionError("line image not wide enough (segmentation error?)",image=image)
        if image.shape[1]>10000:
            raise common.RecognitionError("line image too wide???",image=image)

        # FIXME for some reason, something down below
        # depends on this being a bytearray image, so
        # we're normalizing it here to that type
        image = array(image*255.0/amax(image),'B')

        # compute the raw segmentation
        rseg = self.segmenter.charseg(image)
        # if self.display:
        # show_segmentation(rseg) # FIXME
        rseg = renumber_labels(rseg,1) # FIXME
        if amax(rseg)<self.minsegs: 
            raise common.RecognitionError("not enough segments in raw segmentation",rseg=rseg)
        # self.grouper = grouper.Grouper()
        pseg = self.segmenter0.charseg(image)
        self.grouper.setSegmentation(rseg,preferred=pseg)

        # compute the geometry (might have to use
        # CCS segmenter if this doesn't work well)
        geo = docproc.seg_geometry(rseg)
        mheight = geo[0]
        self.mheight = mheight

        # invert the input image (make a copy first)
        old = image
        image = amax(image)-image

        # initialize the whitespace estimator
        self.whitespace.setLine(image,rseg)
        
        # this holds the list of recognized characters if keep!=0
        self.chars = []
        
        # debugging display
        if self.display:
            clf()
            gray()

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
                raise common.RecognitionError("bad line geometry",geo=geo)
            ry,rw,rh = rel
            assert rw>0 and rh>0,"error: rw=%g rh=%g"%(rw,rh)
            rel = docproc.rel_geo_normalize(rel)

            # extract the character image (and optionally display it)
            (raw,mask) = self.grouper.extractWithMask(image,i,1)
            char = raw/255.0

            # Add a skip transition with the pixel width as cost.
            # This ensures that the lattice is at least connected.
            # Note that for typical character widths, this is going
            # to be much larger than any per-charcter cost.
            if self.add_rho:
                self.grouper.setClass(i,"_",self.rho_scale*raw.shape[1])

            # compute the classifier output for this character
            # FIXME parallelize this
            outputs = self.cmodel.coutputs(char,geometry=rel)
            assert len(set(map(lambda x:x[0],outputs)))==len(outputs),"classifier outputs contains repeated classes"
            outputs = [(x[0],-log(x[1])) for x in outputs]
            # print "@@@",i,self.grouper.getSegments(i),outputs[:2]

            # estimate the space cost
            sc = self.whitespace.classifySpace(x1)
            yes_space = min(self.maxspacecost,-log(sc[1]))
            no_space = min(self.maxspacecost,-log(sc[0]))

            # maybe add a transition on "_" that we can use to skip 
            # this character if the transcription contains a "~"
            if not self.norejects:
                self.grouper.setClass(i,"~",self.reject_cost)

            # extra penalty based on segmentation
            # (right now, it's only for combining characters, but
            # we may add costs for splitting too)
            segcost = 0.0
            if self.grouper.isCombined(i):
                if self.combined_cost>0.0:
                    segcost += self.combined_cost
            elif self.grouper.isSplit(i):
                if self.split_cost>0.0:
                    segcost += self.split_cost
            
            if self.debug_cls is not None:
                matching = [k for k,v in outputs[:int(self.nbest)] if re.match(self.debug_cls,k)]
                if len(matching)>0:
                    print "grouper","%3d"%i,
                    print "(%3d,%3d)"%char.shape,
                    print "   (y %5.2f w %5.2f h %5.2f)"%(rel[0],rel[1],rel[2]),
                    print "   %6.2f+"%segcost,
                    print "%-15s"%("-".join([str(x) for x in self.grouper.getSegments(i)])),
                    for c,v in outputs[:5]: print "%s_%.2f"%(c,v),
                    print

            # add the top classes to the lattice
            outputs.sort(key=lambda x:x[1])
            for cls,cost in outputs[:int(self.nbest)]:
                # don't add anything with a cost above maxcost
                # if cost>self.maxcost and cls!="~": continue

                # add rejects only if there is nothing else
                if self.norejects and cls=="~" and len(outputs)>1:
                    continue

                if cls=="":
                    print "warning: empty class label from classifier"
                    cls = "~"

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

                if type(cls)==int:
                    assert self.allow_any or (cls>=0 and cls<0x110000),\
                        "classifier returned non-unicode class: %s"%(hex(cls),)
                elif type(cls)==str:
                    assert len(cls)<4,\
                        ("classifier returned too many chars: %s",cls)
                # for anything else, just add the classified character to the grouper
                if type(cls)==str or type(cls)==unicode:
                    self.grouper.setClass(i,cls,cost+segcost)
                elif type(cls)==int:
                    assert cls>=0 and cls<0x110000,"bad class: %s"%(hex(cls),)
                    self.grouper.setClass(i,cls,cost)
                else:
                    raise Exception("bad class type: %s"%type(cls))
                if self.spaces:
                    self.grouper.setSpaceCost(i,float(yes_space),float(no_space))

            # display it for debugging purposes
            if self.display:
                cls,cost = outputs[0]
                subplot(self.display_shape[0],self.display_shape[1],1+i%prod(self.display_shape))
                gca().set_frame_on(False)
                if cost>0.2: ylabel("%d"%int(10*cost),color='red',size=10)
                if self.grouper.isCombined(i): l = "*"
                elif self.grouper.isSplit(i): l= "-"
                else: l = " "
                xlabel("%d%s %s"%(i,l,cls),color='blue',size=10)
                xticks([])
                yticks([])
                imshow(char,interpolation='nearest'); ginput(1,0.001)
                if (i+1)%prod(self.display_shape)==0:
                    ginput(1,10000)
                    clf()
                    ginput(1,0.001)
                

            # record information for debugging
            self.chars.append(common.Record(index=i,
                                            image=char,
                                            outputs=outputs,
                                            segcost=segcost,
                                            comb=self.grouper.isCombined(i),
                                            split=self.grouper.isSplit(i)))
        self.rseg = rseg

    def bestpath(self):
        """Return the bestpath through the recognition lattice, as a string.
        This is used for debugging."""
        return self.grouper.bestpath()

    def getLattice(self):
        return self.grouper
    
    def saveLattice(self,stream):
        """Save the recognition lattice to a file."""
        self.grouper.saveLattice(stream)

    def save(self,file):
        with open(file,"w") as stream:
            pickle.dump(self,stream)
    def load(self,file):
        with open(file,"r") as stream:
            obj = pickle.load(self,stream)
        self.__dict__.update(obj.__dict__)

import mlp

class MlpModel(ClassifierModel):
    makeClassifier = mlp.MLP
    makeExtractor = BboxFE
    def __init__(self,**kw):
        ClassifierModel.__init__(self,**kw)
    def name(self):
        return str(self)
    def setExtractor(self,e):
        pass

class AutoMlpModel(ClassifierModel):
    makeClassifier = mlp.AutoMLP
    makeExtractor = BboxFE
    def __init__(self,**kw):
        ClassifierModel.__init__(self,**kw)
    def name(self):
        return str(self)
    def setExtractor(self,e):
        pass
