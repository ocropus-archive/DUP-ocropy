import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology

import iulib,ocropus
import utils
from utils import allsplitext,write_text
import docproc
import ligatures
import fstutils
import openfst
import ocrorast
import ocrolseg
import ocropreproc
import common

import cPickle as pickle
pickle_mode = 2

class OcroFST():
    def __init__(self,native=None):
        if native:
            assert isinstance(native,ocropus.OcroFST)
            self.comp = native
        else:
            self.comp = ocropus.make_OcroFST()
    def clear(self):
        self.comp.clear()
    def newState(self):
        return self.comp.newState()
    def addTransition(self,frm,to,output,cost=0.0,inpt=None):
        if inpt is None:
            self.comp.addTransition(frm,to,output,cost)
        else:
            self.comp.addTransition(frm,to,output,cost,inpt)
    def setStart(self,node):
        self.comp.setStart(node)
    def setAccept(self,node,cost=0.0):
        self.comp.setAccept(node,cost)
    def special(self,s):
        return self.comp.special(s)
    def bestpath(self):
        result = iulib.ustrg()
        self.comp.bestpath(result)
        return common.ustrg2unicode(result)
    def setString(self,s,costs,ids):
        self.comp.setString(unicode2ustrg(s),costs,ids)
    def nStates(self):
        return self.comp.nStates()
    def getStart(self):
        return self.comp.getStart()
    def getAcceptCost(self,node):
        return self.comp.getAcceptCost(node)
    def isAccepting(self,node):
        return self.comp.isAccepting(node)
    def getTransitions(self,frm):
        tos = iulib.intarray()
        symbols = iulib.intarray()
        costs = iulib.floatarray()
        inputs = iulib.intarray()
        self.comp.getTransitions(tos,symbols,costs,inputs,frm)
        return (iulib.numpy(tos,'i'),iulib.numpy(symbols,'i'),iulib.numpy(costs),iulib.numpy(inputs,'i'))
    def rescore(frm,to,symbol,new_cost,inpt=None):
        if inpt is None:
            self.comp.resocre(frm,to,symbol,new_cost)
        else:
            self.comp.resocre(frm,to,symbol,new_cost,inpt)
    def load(self,name):
        self.comp.load(name)
        return self
    def save(self,name):
        self.comp.save(name)
    def as_openfst(self):
        import openfst,os
        tmpfile = "/tmp/%d.fst"%os.getpid()
        self.comp.save(tmpfile)
        fst = openfst.StdVectorFst()
        fst.Read(tmpfile)
        os.unlink(tmpfile)
        return fst

def native_fst(fst):
    if isinstance(fst,openfst.StdVectorFst):
        fst = fstutils.openfst2ocrofst(fst)
    if isinstance(fst,ocropus.OcroFST): 
        return fst
    if isinstance(fst,OcroFST): 
        assert isinstance(fst.comp,ocropus.OcroFST)
        return fst.comp
    raise Exception("expected either native or Python FST, got %s"%fst)

### native code beam search

def beam_search_simple(u,v,n):
    """Perform a simple beam search through the two lattices; beam width is
    given by n."""
    s = iulib.ustrg()
    cost = ocropus.beam_search(s,native_fst(u),native_fst(v),n)
    return common.ustrg2unicode(s),cost

def beam_search(lattice,lmodel,beam):
    """Perform a beam search through the lattice and language model, given the
    beam size.  Returns (v1,v2,input_symbols,output_symbols,costs)."""
    v1 = iulib.intarray()
    v2 = iulib.intarray()
    ins = iulib.intarray()
    outs = iulib.intarray()
    costs = iulib.floatarray()
    ocropus.beam_search(v1,v2,ins,outs,costs,native_fst(lattice),native_fst(lmodel),beam)
    return (iulib.numpy(v1,'i'),iulib.numpy(v2,'i'),iulib.numpy(ins,'i'),
            iulib.numpy(outs,'i'),iulib.numpy(costs,'f'))

