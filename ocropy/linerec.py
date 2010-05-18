import sys,os,re,glob,math,glob,signal,cPickle
import iulib,ocropus
import components
from utils import N,NI,F,FI,Record,show_segmentation
from scipy.ndimage import interpolation
from pylab import *
import unicodedata
import pickle
import __init__ as ocropy
import fstutils

def bestpath(lattice):
    s = ocropy.ustrg()
    lattice.bestpath(s)
    cost = 0.0
    return ocropy.ustrg_as_string(s)

def rect(r):
    return (r.x0,r.y0,r.x1,r.y1)

def floatimage(image):
    fimage = ocropy.floatarray()
    fimage.copy(image)
    ocropy.div(fimage,255.0)
    return fimage

class RejectException(Exception):
    def __init__(self,*args,**kw):
        super.__init__(*args,**kw)

class LineRecognizer: # can't inherit -- breaks save/load (ocropus.IRecognizeLine):
    def __init__(self,cmodel=None,segmenter="DpSegmenter",best=10,
                 maxcost=10.0,reject_cost=100.0,minheight_letters=0.5):
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
        print "[Python LineRecognizer]"
        self.set_defaults()
        self.cmodel = cmodel
        self.best = best
        self.maxcost = maxcost
        self.reject_cost = reject_cost
        self.min_height = minheight_letters

    def set_defaults(self):
        self.debug = 0
        self.segmenter = components.make_ISegmentLine("DpSegmenter")
        self.grouper = components.make_IGrouper("SimpleGrouper")
        self.cmodel = None
        self.best = 10
        self.maxcost = 10.0
        self.reject_cost = 100.0
        self.min_height = 0.5
        self.rho_scale = 1.0
        self.maxoverlap = 0.8

    def info(self):
        for k in sorted(self.__dict__.keys()):
            print k,self.__dict__[k]
        self.cmodel.info()

    def recognizeLine(self,lattice,image):
        "Recognize a line, outputting a recognition lattice."""
        rseg = iulib.intarray()
        return self.recognizeLineSeg(lattice,rseg,image)

    def recognizeLineSeg(self,lattice,rseg,image,keep=0):
        """Recognize a line.
        lattice: result of recognition
        rseg: intarray where the raw segmentation will be put
        image: line image to be recognized"""

        ## compute the raw segmentation
        self.segmenter.charseg(rseg,image)
        ocropus.make_line_segmentation_black(rseg)
        if self.debug: show_segmentation(rseg)
        iulib.renumber_labels(rseg,1)
        self.grouper.setSegmentation(rseg)

        # compute the median segment height
        heights = []
        for i in range(self.grouper.length()):
            bbox = self.grouper.boundingBox(i)
            heights.append(bbox.height())
        mheight = median(array(heights))
        self.mheight = mheight

        # invert the input image (make a copy first)
        old = image; image = iulib.bytearray(); image.copy(old)
        iulib.sub(255,image)

        # allocate working arrays
        segs = iulib.intarray()
        raw = iulib.bytearray()
        mask = iulib.bytearray()

        # this holds the list of recognized characters if keep!=0
        self.chars = []
        
        # now iterate through the characters
        for i in range(self.grouper.length()):
            # get the bounding box for the character (used later)
            bbox = self.grouper.boundingBox(i)
            aspect = bbox.height()*1.0/bbox.width()

            # extract the character image (and optionally display it)
            self.grouper.extractWithMask(raw,mask,image,i,1)
            char = NI(raw)
            char = char / float(amax(char))
            if self.debug:
                imshow(char)
                raw_input()

            # Add a skip transition with the pixel width as cost.
            # This ensures that the lattice is at least connected.
            # Note that for typical character widths, this is going
            # to be much larger than any per-charcter cost.
            self.grouper.setClass(i,ocropus.L_RHO,self.rho_scale*raw.dim(0))

            # compute the classifier output for this character
            # print self.cmodel.info()
            outputs = self.cmodel.coutputs(FI(char))
            outputs = [(x[0],-log(x[1])) for x in outputs]
            if keep:
                self.chars.append(Record(index=i,image=char,outputs=outputs))
            
            # add the top classes to the lattice
            outputs.sort(key=lambda x:x[1])
            s = iulib.ustrg()
            for cls,cost in outputs[:self.best]:
                # don't add the reject class (written as "~")
                if cls=="~": continue

                # letters are never small, so we skip small bounding boxes that
                # are categorized as letters; this is an ugly special case, but
                # it is quite common
                ucls = cls
                if type(cls)==str: ucls = unicode(cls,"utf-8")
                category = unicodedata.category(ucls[0])
                if bbox.height()<self.min_height*mheight and category[0]=="L":
                    # add an empty transition to allow skipping junk
                    # (commented out right now because I'm not sure whether
                    # the grouper can handle it; FIXME)
                    # self.grouper.setClass(i,"",1.0)
                    continue

                # for anything else, just add the classified character to the grouper
                s.assign(cls)
                self.grouper.setClass(i,s,min(cost,self.maxcost))
                # FIXME better space handling
                self.grouper.setSpaceCost(i,0.5,0.0)

        # extract the recognition lattice from the grouper
        self.grouper.getLattice(lattice)

        # return the raw segmentation as a result
        return rseg

    def startTraining(self,type="adaptation",model="LatinClassifier"):
        """Instantiate a new character model, plus size and space models."""
        self.new_model = ocropy.make_IModel(model)
        self.new_model.pset("cds","bitdataset")
        self.new_model.setExtractor("StandardExtractor")

    def finishTraining(self):
        """After all the data has been loaded, this performs the actual training."""
        self.new_model.updateModel()
        self.cmodel = self.new_model
        self.new_model = None

    def addTrainingLine1(self,image,transcription):
        """Add a line of text plus its transcription to the line recognizer as
        training data."""
        self.addTrainingLine(ocropy.intarray(),image,transcription)

    def addTrainingLine(self,rseg,image_,transcription):
        """Add a line of text plus its transcription to the line recognizer as
        training data. This also returns the raw segmentation in its first
        argument (an intarray)."""
        # rseg = ocropy.intarray()

        # make a copy of the input image
        image = ocropy.bytearray()
        image.copy(image_)

        # now run the recognizer
        lattice = ocropy.make_OcroFST()
        self.recognizeLineSeg(lattice,rseg,image)
        print "bestpath",bestpath(lattice)

        # compute the alignment
        print "gt",transcription
        lmodel = fstutils.make_line_fst([transcription])
        r = ocropy.compute_alignment(lattice,rseg,lmodel)
        result = r.output
        cseg = r.cseg
        costs = r.costs.numpy()
        tcost = sum(costs)
        if tcost>10000.0: raise Exception("cost too high")
        mcost = mean(costs)
        if mcost>10.0: raise RejectException("mean cost too high")
        if tcost>100.0: raise RejectException("total cost too high")
        print "alignment",mcost,tcost

        # this is a special case of ocropus-extract-csegs...

        # find all the aligned characters
        ocropy.sub(255,image)
        utext = ocropy.ustrg()
        # utext.assign(text)
        utext.assign(r.output)
        self.grouper.setSegmentationAndGt(rseg,cseg,utext)
        chars = []
        for i in range(self.grouper.length()):
            cls = self.grouper.getGtClass(i)
            if cls==-1: continue # ignore missegmented characters (handled separately below)
            cls = chr(cls)
            raw = ocropy.bytearray()
            mask = ocropy.bytearray()
            self.grouper.extractWithMask(raw,mask,image,i,1)
            chars.append(Record(raw=raw,mask=mask,cls=cls,index=i,bbox=self.grouper.boundingBox(i)))

        # find all the non-aligned groups and add them as nonchars
        bboxes = [rect(c.bbox) for c in chars]
        self.grouper.setSegmentation(rseg)
        nonchars = []
        for i in range(self.grouper.length()):
            bbox = self.grouper.boundingBox(i)
            fractions = [min(bbox.fraction_covered_by(c.bbox),c.bbox.fraction_covered_by(bbox)) for c in chars]
            covered = max(fractions)
            assert covered>1e-5
            if covered>self.maxoverlap: continue
            assert rect(bbox) not in bboxes
            raw = ocropy.bytearray()
            mask = ocropy.bytearray()
            self.grouper.extractWithMask(raw,mask,image,i,1)
            nonchars.append(Record(raw=raw,mask=mask,cls=cls,index=i,bbox=self.grouper.boundingBox(i)))

        # finally add them to the character model
        for c in chars:
            self.new_model.cadd(floatimage(c.raw),c.cls)
        for c in nonchars:
            self.new_model.cadd(floatimage(c.raw),"~")
        print "#chars",len(chars),"#nonchars",len(nonchars)

    def align(self,chars,cseg,costs,image,transcription):
        """Align an image with its transcription, returning the characters, cseg,
        and per-character costs."""
        lattice = ocropy.make_OcroFST()
        self.recognizeLineSeg(lattice,rseg,image)
        print "bestpath",bestpath(lattice)
        lmodel = self.makeLineModel(transcription)
        r = ocropy.compute_alignment(lattice,rseg,lmodel)
        result = r.output
        costs.copy(r.costs)
        chars.clear() # FIXME
        raise Exception("unimplemented")

    def makeLineModel(self,s):
        raise Exception("unimplemented")

    def epoch(self,n):
        """For stochastic gradient descent, models are often trained for multiple
        epochs.  Method is used to notify this class that a new training epoch has
        started. It is just passed on to the model."""
        if hasattr(self.cmodel,"epoch"):
            self.cmodel.epoch(n)

    def save(self,file):
        """Save the line recognizer as a .pymodel (the file name must
        end in .pymodel)."""
        assert ".pymodel" in file
        with open(file,"w") as stream:
            pickle.dump(self,stream)

    def load(self,file):
        """Load a line recognizer.  This handles both a .cmodel and a .pymodel
        file."""
        self.set_defaults()
        if ".pymodel" in file:
            with open(file,"r") as stream:
                obj = pickle.load(self,stream)
            for k,v in obj.__dict__:
                self.__dict__[k] = v
        elif ".cmodel" in file:
            self.cmodel = ocropy.load_IModel(file)
        else:
            raise Exception("unknown extension")
