import sys,os,re,glob,math,glob,signal
import iulib,ocropus
import components
from utils import N,NI,F,FI,Record,show_segmentation
from scipy.ndimage import interpolation
from pylab import *
import unicodedata
import pickle

class CmodelLineRecognizer:
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
        self.debug = 0
        self.segmenter = components.make_ISegmentLine(segmenter)
        self.grouper = components.make_IGrouper("SimpleGrouper")
        # self.grouper.pset("maxdist",5)
        # self.grouper.pset("maxrange",5)
        self.cmodel = cmodel
        self.best = best
        self.maxcost = maxcost
        self.reject_cost = reject_cost
        self.min_height = minheight_letters
        self.rho_scale = 1.0

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

    def startTraining(self,type="adaptation"):
        raise Exception("unimplemented")
    def finishTraining(self):
        raise Exception("unimplemented")
    def addTrainingLine(self,image,transcription):
        raise Exception("unimplemented")
    def addTrainingLine(self,segmentation,image,transcription):
        raise Exception("unimplemented")
    def align(self,chars,seg,costs,image,transcription):
        raise Exception("unimplemented")
    def epoch(self,n):
        raise Exception("unimplemented")
    def save(self,file):
        assert ".pymodel" in file
        with open(file,"w") as stream:
            pickle.dump(self,stream)
    def load(self,file):
        assert ".pymodel" in file
        with open(file,"r") as stream:
            obj = pickle.load(self,stream)
        for k,v in obj.__dict__:
            self.__dict__[k] = v
