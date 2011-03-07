# -*- encoding: utf-8 -*-

import sys,os,re,codecs
import openfst
import ocropus,iulib
import common
import ligatures

epsilon = openfst.epsilon
sigma = ocropus.L_RHO
space = ligatures.lig.ord(" ")
reject = ligatures.lig.ord("~")

Fst = openfst.StdVectorFst

def add_string(fst,start,end,s,cost=0.0):
    for i in range(len(s)):
        c = ord(s[i])
        next = fst.AddState() if i<len(s)-1 else end
        fst.AddArc(start,c,c,cost if i==0 else 0.0,next)
        start = next

class STR:
    """Add a string to an FST."""
    def __init__(self,s,cost=0.0):
        self.s = s
        self.cost = cost
    def generate(self,fst,start,end):
        add_string(fst,start,end,self.s,self.cost)

def asgen(s):
    """Coerce a string to an FST"""
    if type(s)==str or type(s)==unicode:
        return STR(s)
    else:
        return s

class SEQ:
    """A sequence of FSTs.  Example: SEQ("AB","C","DEF")==STR("ABCDEF")."""
    def __init__(self,*args):
        self.args = args
    def generate(self,fst,start,end):
        args = self.args
        for i in range(len(args)):
            next = fst.AddState() if i<len(args)-1 else end
            asgen(args[i]).generate(fst,start,next)
            start = next

class Y:
    """A range of characters, as in Y("A-Za-z,.!0-9"), similar to regular expressions."""
    def __init__(self,s,cost=0.0,condition=lambda x:True):
        self.s = s
        self.cost = cost
    def generate(self,fst,start,end):
        s = self.s
        i = 0
        while i<len(s):
            lo = s[i]
            hi = s[i]
            if i<len(s)-2 and s[i+1]=="-":
                hi = s[i+2]
                i += 3
            else:
                i += 1
            for c in range(ord(lo),ord(hi)+1):
                if condition(unichr(c)):
                    fst.AddArc(start,c,c,self.cost,end)

DIGITS = Y("0-9")
LETTER = Y("A-Za-z")
WS = Y(" \t\n")
ASCII = Y("@-~")

class RANGE:
    """A range of characters, as in RANGE("A","Z")."""
    def __init__(self,lo,hi,cost=0.0):
        self.lo = lo
        self.hi = hi
        self.cost = cost
    def generate(self,fst,start,end):
        for c in range(ord(self.lo),ord(self.hi)+1):
            fst.AddArc(start,c,c,self.cost,end)

class ALT:
    """An set of alternatives.  ALT("A","B","C") is one of
    the three strings, written as A|B|C in regular expressions."""
    def __init__(self,*args):
        self.args = args
        self.cost = 0.0
    def generate(self,fst,start,end):
        for arg in self.args:
            if type(arg)==str or type(arg)==unicode:
                arg = STR(arg,cost=cost)
            asgen(arg).generate(fst,start,end)

class STAR:
    """Zero or more repetitions.  STAR("A") is the analog
    of A* in regular expressions."""
    def __init__(self,expr,cost=0.0):
        self.expr = expr
        self.cost = cost
    def generate(self,fst,start,end):
        start2 = fst.AddState()
        fst.AddArc(start,epsilon,epsilon,0.0,start2)
        fst.AddArc(start2,epsilon,epsilon,0.0,end)
        asgen(self.expr).generate(fst,start2,star2)

class PLUS:
    """One or more repetitions.  PLUS("A") is the analog
    of A+ in regular expressions."""
    def __init__(self,expr,cost=0.0):
        self.expr = expr
        self.cost = cost
    def generate(self,fst,start,end):
        start2 = fst.AddState()
        end2 = fst.AddState()
        fst.AddArc(start,epsilon,epsilon,0.0,start2)
        fst.AddArc(end2,epsilon,epsilon,0.0,end)
        asgen(self.expr).generate(fst,start2,end2)
        fst.AddArc(end2,epsilon,epsilon,0.0,start2)

class DICTIONARY:
    """Load a dictionary and add it to the FST.
    Equivalent to ALT(STR("line1"),STR("line2"),...)
    for the lines of the file.  If the file contains lines
    of the form "cost\ttext\n", then the cost is associated
    with the line, otherwise the cost is 0."""
    def __init__(self,fname):
        self.fname = fname
    def generate(self,fst,start,end):
        with open(fname) as stream:
            for line in stream.readlines():
                cost = 0.0
                line = line[:-1]
                # TAB indicates the presence of costs
                if "\t" in line: cost,line = line.split("\t")
                add_string(fst,start,end,line)

class LOADFST:
    """Load an existing FST and splice it into this FST."""
    def __init__(self,fname):
        raise Exception("unimplemented")
    def generate(self,fst,start,end):
        raise Exception("unimplemented")

def GEN(expr):
    """Generate the FST described by the expression and return it."""
    fst = openfst.StdVectorFst()
    start = fst.AddState()
    fst.SetStart(start)
    end = fst.AddState()
    fst.SetFinal(end,0.0)
    expr.generate(fst,start,end)
    return fst

fst = GEN(PLUS(ALT(PLUS("A"),PLUS("B"))))
fst.Write("test.fst")

