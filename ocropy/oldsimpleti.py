import signal
signal.signal(signal.SIGINT,lambda *args:sys.exit(1))
from pylab import *
import sys,os,re,ocropy,optparse
from ocropy import N,NI
from scipy.ndimage import measurements
from scipy.misc import imsave
from PIL import Image

def closure(temp,w_threshold=0,h_threshold=0):
    w,h = (temp.dim(0),temp.dim(1))
    components = ocropy.intarray()
    components.copy(temp)
    n = ocropy.label_components(components)
    boxes = ocropy.rectarray()
    ocropy.bounding_boxes(boxes,components)
    result = ocropy.bytearray()
    result.resize(w,h)
    result.fill(0)
    for i in range(1,boxes.length()):
        r = boxes.at(i)
        if r.width()<w_threshold and r.height()<h_threshold: continue
        ocropy.fill_rect(result,r,255)
    return result

class SimpleTIClassification(ocropy.ITextImageClassification):
    def __init__(self):
        # print "SimpleTIClassification instantiated"
        pass
    def interface(self):
        return "ITextImageClassification"
    def textImageMap(self,image,od=0,cd=0):
        w,h = (image.dim(0),image.dim(1))
        binarizer = ocropy.make_IBinarize("BinarizeBySauvola")
        binarizer.pset("k",0.2)
        binarizer.pset("w",200)
        bin = ocropy.bytearray()
        binarizer.binarize(bin,image)
        ocropy.binary_invert(bin)
        if od>0:
            ocropy.binary_open_rect(bin,od,od)
        if cd>0:
            ocropy.binary_close_rect(bin,cd,cd)
        result = closure(bin,100,100)
        result = closure(result,400,300)
        result = closure(result,400,300)
        return result
    def textImageProbabilities(self,rgb,input,od=0,cd=0):
        w,h = (input.dim(0),input.dim(1))
        map = self.textImageMap(input,od=od,cd=cd)
        rmap = ocropy.bytearray(w,h)
        rmap.copy(map)
        ocropy.sub(255,rmap)
        zeros = ocropy.bytearray(w,h)
        zeros.fill(0)
        ocropy.pack_rgb(rgb,rmap,map,zeros)
        # rgb.copy(map)
        # ocropy.mul(rgb,256)


