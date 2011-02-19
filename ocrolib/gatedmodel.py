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

class GatedModel:
    def __init__(self):
        self.center_extractor = ocrolib.BboxFE()
        self.gates = []
        self.models = []
    def extract(self,image):
        v = self.center_extractor.extract(image)
        v /= sqrt(sum(v**2))
        return v
    def addGated(self,gate,model):
        self.gates.append(gate)
        self.models.append(model)
    def coutputs(self,image,geometry=None):
        outputs = []
        v = self.extract(image)
        outputs = []
        for i in range(len(self.gates)):
            if not self.gates[i].check(v): continue
            output = self.models[i].coutputs(image,geometry=geometry)
            outputs.append(output)
        result = average_outputs(outputs)
        result.sort(key=lambda x:x[1],reverse=1)
        return result
    def cclassify(self,image,geometry=None):
        outputs = self.coutputs(image,geometry)
        if len(outputs)<1: return None
        return outputs[0][0]
    def coutputs_batch(self,vs,geometries=None):
        outputs = []
        for i in range(len(vs)):
            v = vs[i]
            geometry = None
            if geometries is not None: geometry = geometries[i]
            outputs.append(self.coutputs(v,geometry=geometry))
        return outputs
    def setExtractor(self,extractor):
        ocrolib.warn_once("setExtractor called")

class AlwaysGate:
    def check(self,v):
        return True

class DistanceGate:
    def __init__(self,center,cutoff):
        self.center = center
        self.cutoff = cutoff
    def check(self,v):
        return quant.dist(self.center.ravel(),v.ravel())<=self.cutoff

def load_gatedmodel(gated,args,cutoff=0.75):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            paths += glob.glob(arg+"/*.cmodel") + glob.glob(arg+"/*.model") 
        else:
            paths.append(arg)
    for path in paths:
        print "# loading",path
        model = ocrolib.load_component(path)
        base,_ = os.path.splitext(path)
        if os.path.exists(base+".info"):
            with open(base+".info","r") as stream:
                info = cPickle.load(stream)
            bins = len(info.cutoffs)
            d = info.cutoffs[int(bins*cutoff)]
            gated.addGated(DistanceGate(info.center,d),model)
        else:
            gated.addGated(AlwaysGate(),model)
    return gated

