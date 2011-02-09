#!/usr/bin/python
import random as pyrandom
import code,pickle,sys,os,re,traceback,cPickle,os.path,glob
from optparse import OptionParser
from pylab import *
from scipy import stats
import common as ocrolib
import dbtables,quant,utils

def average_outputs(os):
    result = {}
    for o in os:
        for k,v in o:
            result[k] = result.get(k,0.0)+v
    n = 1.0*len(os)
    return [(k,v/n) for k,v in result.items()]

class MultiClass:
    def __init__(self):
        self.cutoff = 0.75
        self.extractor = ocrolib.ScaledFE()
        self.loadDir("/home/tmb/f/multiclass/unlv-multi-model/")
    def extract(self,image):
        v = self.extractor.extract(image)
        v /= sqrt(sum(v**2))
        return v
    def setExtractor(self,extractor):
        pass
    def loadDir(self,dir):
        self.vectors = []
        self.cutoffs = []
        self.models = []
        paths = glob.glob(dir+"/*.cmodel") + glob.glob(dir+"/*.model") 
        print "# loading",len(paths),"models"
        for path in paths:
            model = ocrolib.load_component(path)
            base,_ = os.path.splitext(path)
            if os.path.exists(base+".info"):
                with open(base+".info","r") as stream:
                    info = cPickle.load(stream)
                center = info.center
                cutoff = info.cutoffs[int(len(info.cutoffs)*self.cutoff)]
            else:
                center = None
                cutoff = None
            print "#",path,cutoff
            self.vectors.append(center)
            self.cutoffs.append(cutoff)
            self.models.append(model)
        self.cutoffs = array(self.cutoffs,'f')
    def coutputs(self,v,geometry=None):
        outputs = []
        v = self.extract(v)
        dists = array([(v and quant.dist(v,c)) or 0.0 for c in self.vectors],'f')
        for i in range(len(dists)):
            if dists[i]<self.cutoffs[i]:
                va = concatenate([v.ravel(),array(geometry,'f')])
                outputs.append(self.models[i].coutputs(va,geometry=geometry))
        # print "averaging",len(outputs)
        return average_outputs(outputs)
