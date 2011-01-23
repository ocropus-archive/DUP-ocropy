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
        self.extractor = ocrolib.ScaledFE()
        self.base_model = ocrolib.load_component("/home/tmb/f/models/2m2-reject.cmodel")
        self.load("/home/tmb/f/ocropy/unlv-multi-model/")
    def extract(self,v):
        v /= sqrt(sum(v**2))
        v = self.extractor.extract(v)
        return v
    def load(self,dir):
        self.vectors = []
        self.cutoffs = []
        self.models = []
        paths = glob.glob(dir+"/*.model")
        print "# loading",len(paths),"models"
        for path in paths:
            print "#",path
            model = ocrolib.load_component(path)
            base,_ = os.path.splitext(path)
            with open(base+".info","r") as stream:
                info = cPickle.load(stream)
            self.vectors.append(info.center)
            self.cutoffs.append(info.cutoff2)
            self.models.append(model)
        self.cutoffs = array(self.cutoffs,'f')
    def setExtractor(self,extractor):
        pass
    def coutputs(self,v,geometry=None):
        outputs = []
        outputs.append(self.base_model.coutputs(v,geometry))
        v = self.extract(v)
        dists = array([quant.dist(v,c) for c in self.vectors],'f')
        for i in range(len(dists)):
            if dists[i]<self.cutoffs[i]:
                va = concatenate([v.ravel(),array(geometry,'f')])
                outputs.append(self.models[i].coutputs(va))
        # print "averaging",len(outputs)
        return average_outputs(outputs)
        
