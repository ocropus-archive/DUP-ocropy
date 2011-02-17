#!/usr/bin/python
import code,pickle,sys,os,re,traceback
from optparse import OptionParser
from pylab import *
import common as ocrolib
import dbtables,docproc,mlp,utils
from scipy.ndimage import interpolation

display_training = 0
display_cls = 0

if display_training or display_cls:
    ion(); gray()

class BadImage(Exception):
    pass
class BadGroundTruth(Exception):
    pass

class WhitespaceModel:
    def __init__(self):
        self.s = 20
        self.r = 2
    def showLine(self):
        mh,a,b = self.line_params
        if display_training:
            clf()
            imshow(self.image)
            h,w = self.image.shape
            draw()
            plot([0,w],[b,a*w+b],"r-")
            raw_input()
            clf()
    def setLine(self,image,cseg=None):
        if amax(image)<1e-6: raise BadImage()
        self.image = array(image*(1.0/amax(image)),'f')
        self.cseg = cseg
        self.line_params = docproc.seg_geometry(cseg,math=0)
    def getSubImage(self,x):
        r = self.r
        image = self.image
        mh,a,b = self.line_params
        dy = int(mh)
        dx = int(r*mh+0.5)
        yc = int(a*x+b)
        sub = docproc.extract(image,(yc-dy,x-dx,yc+dy,x+dx))
        sub = interpolation.zoom(sub,(self.s+0.2)*(1.0/sub.shape[0]),order=1)
        sub = sub[:self.s,:self.r*self.s]
        return sub
    def classifySpace(self,x):
        image = self.getSubImage(x)
        sub = image.ravel()
        result = self.mlp.outputs(sub.reshape(1,len(sub)))[0]
        if display_cls:
            print result
            clf(); imshow(image); draw()
            raw_input()
        return result
    def startTraining(self):
        self.data = []
        self.classes = []
    def csegWhiteIterator(self,image,cseg,gt):
        assert image.shape==cseg.shape
        self.setLine(image,cseg=cseg)
        self.showLine()
        h,w = image.shape
        geo = docproc.seg_geometry(cseg)
        grouper = ocrolib.StandardGrouper()
        grouper.pset("maxrange",1)
        grouper.setSegmentation(cseg)
        if len(gt)!=grouper.length():
            raise BadGroundTruth()
        s = None
        for i in range(grouper.length()-1):
            bbox = grouper.boundingBox(i)
            y0,x0,y1,x1 = bbox
            sub = self.getSubImage(x1)
            cls = gt[i]
            if cls==" ": continue
            spc = (gt[i+1]==" ")
            yield (sub,spc,cls)
    def trainLineAndCseg(self,image,cseg,gt):
        assert image.shape==cseg.shape
        assert image.dtype==dtype('B')
        assert cseg.dtype==dtype('i')
        items = list(self.csegWhiteIterator(image,cseg,gt))
        for sub,spc,cls in items:
            if display_training:
                print spc,cls
                clf(); imshow(sub); draw()
                raw_input()
            if amax(sub)<1e-6: raise BadImage()
            sub = sub*(1.0/amax(sub))
            assert sub.size==self.r*self.s*self.s, \
                "unexpected size: %s (after %d samples)"%(sub.shape,len(self.data))
            self.data.append(sub.ravel())
            self.classes.append(int(spc))
    def updateModel(self):
        data = array(self.data,'float32')
        classes = array(self.classes,'int32')
        print "got",sum(classes==1),"spaces out of",len(self.classes)
        del self.data,self.classes
        self.mlp = mlp.AutoMLP()
        self.mlp.train(data,classes)

