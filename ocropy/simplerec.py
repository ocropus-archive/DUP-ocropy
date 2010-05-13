import sys,os,re,glob,math,glob,signal
import iulib,ocropus
import components
from utils import N,NI,F,FI,Record
from scipy.ndimage import interpolation
from pylab import *

def show_segmentation(rseg):
    temp = iulib.numpy(rseg,type='B')
    temp[temp==255] = 0
    temp = transpose(temp)[::-1,:]
    temp2 = 1 + (temp % 10)
    temp2[temp==0] = 0
    temp = temp2
    print temp.shape,temp.dtype
    temp = temp/float(amax(temp))
    imshow(temp,cmap=cm.spectral); draw()
    raw_input()

class CmodelLineRecognizer:
    def __init__(self,cmodel=None,segmenter="DpSegmenter",best=10,maxcost=10.0,
                 minheight=0.5,minaspect=0.5,maxaspect=20.0):
        self.debug = 0
        self.segmenter = components.make_ISegmentLine(segmenter)
        self.grouper = components.make_IGrouper("SimpleGrouper")
        # self.grouper.pset("maxdist",5)
        # self.grouper.pset("maxrange",5)
        self.cmodel = cmodel
        self.best = best
        self.maxcost = maxcost
        self.min_height = minheight
        self.min_aspect = minaspect
        self.max_aspect = maxaspect

    def recognizeLine(self,lattice,image):
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

        ## compute the median segment height
        heights = []
        for i in range(self.grouper.length()):
            bbox = self.grouper.boundingBox(i)
            heights.append(bbox.height())
        mheight = median(array(heights))
        print "mheight",mheight

        ## now iterate through the characters
        iulib.sub(255,image)
        segs = iulib.intarray()
        raw = iulib.bytearray()
        mask = iulib.bytearray()
        self.chars = []
        for i in range(self.grouper.length()):
            bbox = self.grouper.boundingBox(i)
            if bbox.height()<self.min_height*mheight: continue
            aspect = bbox.height()*1.0/bbox.width()
            if aspect<self.min_aspect: continue
            if aspect>self.max_aspect: continue
            self.grouper.extractWithMask(raw,mask,image,i,1)
            char = NI(raw)
            char = char / float(amax(char))
            if self.debug:
                imshow(char)
                raw_input()

            ## add a skip transition with the pixel width as cost
            self.grouper.setClass(i,ocropus.L_RHO,raw.dim(0))

            ## compute the classifier output for this character
            # print self.cmodel.info()
            outputs = self.cmodel.coutputs(FI(char))
            outputs = [(x[0],-log(x[1])) for x in outputs]
            if keep:
                self.chars.append(Record(index=i,image=char,outputs=outputs))
            
            ## add the top classes to the lattice
            outputs.sort(key=lambda x:x[1])
            s = iulib.ustrg()
            for cls,cost in outputs[:self.best]:
                if cls=="~": continue
                s.assign(cls)
                self.grouper.setClass(i,s,min(cost,self.maxcost))
                self.grouper.setSpaceCost(i,0.5,0.0)

        self.grouper.getLattice(lattice)
        return rseg

    def startTraining(self,type="adaptation"):
        pass
    def finishTraining(self):
        pass
    def addTrainingLine(self,image,transcription):
        pass
    def addTrainingLine(self,segmentation,image,transcription):
        pass
    def align(self,chars,seg,costs,image,transcription):
        pass
    def epoch(self,n):
        pass
    def save(self,file):
        pass
    def load(self,file):
        pass
