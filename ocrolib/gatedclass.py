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

class GatedClass:
    def __init__(self):
        self.center_extractor = ocrolib.BboxFE()
        self.gates = []
        self.models = []
    def extract(self,image):
        v = self.center_extractor.extract(image)
        v /= sqrt(sum(v**2))
        return v
    def add(self,gate,model):
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
        return average_outputs(outputs)
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

def load_gatedclass(gated,dir,cutoff=0.75):
    paths = glob.glob(dir+"/*.cmodel") + glob.glob(dir+"/*.model") 
    print "# loading",len(paths),"models"
    for path in paths:
        model = ocrolib.load_component(path)
        base,_ = os.path.splitext(path)
        if os.path.exists(base+".info"):
            with open(base+".info","r") as stream:
                info = cPickle.load(stream)
            bins = len(info.cutoff)
            d = info.cutoffs[int(bins*cutoff)]
            gated.add(DistanceGate(info.center,d,model))
        else:
            gated.add(AlwaysGate(model))
    return gated

