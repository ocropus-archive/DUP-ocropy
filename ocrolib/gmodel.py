import os,os.path,re,cPickle,numpy
from numpy import *
from common import *
import docproc

def unary(x,lo,hi,steps):
    return array([(t<x) for t in linspace(lo,hi,steps+1)],'f')

class ModelWithGeometry:
    def __init__(self,model="AutoMlpClassifier",extractor="scaledfe",gmode="mag"):
        self.gmode = gmode
        self.model = make_IModel(model)
        self.extractor = make_IExtractor(extractor)
    def nfeatures(self):
        return self.model.nfeatures()
    def nclasses(self):
        return self.model.nclasses()
    def setExtractor(self,extractor):
        self.extractor = make_IExtractor(extractor)
    def updateModel(self):
        self.model.updateModel()
    def augment(self,v,geometry):
        v = self.extractor.extract(v)
        if not self.gmode:
            return v;
        if self.gmode=="mag":
            v = concatenate([v.ravel(),array(geometry,'f')])
            return v
        elif self.gmode=="unary":
            g = array([unary(x,-1,1,20) for x in geometry])
            v = concatenate([v.ravel()]+g)
            return v
        else:
            raise Exception("%s: unknown mode"%self.gmode)
    def cadd(self,v,cls,geometry=None):
        v = self.augment(v,geometry)
        return self.model.cadd(v,cls)
    def coutputs(self,v,geometry=None):
        v = self.augment(v,geometry)
        return self.model.coutputs(v)
    def cclassify(self,v,geometry=None):
        v = self.augment(v,geometry)
        return self.model.cclassify(v)
    def cclassify_par(self,vs,geometries):
        assert len(vs)==len(geometries)
        vs = [self.augment(vs[i],geometries[i]) for i in range(len(vs))]
        outputs = self.model.classify_par(vs)
        return outputs
    def save(self,file):
        with open(file,"w") as stream:
            cPickle.dump(self,stream,2)
    def load(self,file):
        raise Exception("unsupported")
