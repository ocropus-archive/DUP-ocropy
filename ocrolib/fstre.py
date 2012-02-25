# -*- encoding: utf-8 -*-

################################################################
### A simple object-oriented library for constructing FSTs 
### corresponding to regular expressions.
################################################################

import sys,os,re,codecs
import openfst
import iulib
import common
import ligatures

epsilon = openfst.epsilon
sigma = -3 # FIXME

Fst = openfst.StdVectorFst

def add_string(fst,start,end,s,cost=0.0):
    assert type(s)==str or type(s)==unicode
    for i in range(len(s)):
        c = ord(s[i])
        next = fst.AddState() if i<len(s)-1 else end
        fst.AddArc(start,c,c,cost if i==0 else 0.0,next)
        start = next

class STR:
    """Add a string to an FST."""
    def __init__(self,s,cost=0.0):
        assert type(s)==str or type(s)==unicode
        self.s = s
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        add_string(fst,start,end,self.s,self.cost)

def asgen(s):
    """Coerce a string to an FST"""
    if type(s)==str or type(s)==unicode:
        return STR(s)
    else:
        return s

class TRANS:
    """A transduction from string u to string v"""
    def __init__(self,u,v=None,cost=0.0):
        assert type(u)==str or type(u)==unicode
        assert type(v)==str or type(v)==unicode
        self.u = u
        if v is None: v = u
        self.v = v
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        n = max(len(self.u),len(self.v))
        for i in range(n):
            next = fst.AddState() if i<n-1 else end
            l = self.u[i] if i<len(self.u) else epsilon
            l = ord(l) if type(l)==str else l
            m = self.v[i] if i<len(self.v) else epsilon
            m = ord(m) if type(m)==str else m
            c = self.cost if i==0 else 0.0
            fst.AddArc(start,l,m,c,next)
            start = next

class ARC:
    """A single transition with the given input and output labels and cost."""
    def __init__(self,label,olabel=None,cost=0.0):
        assert type(label)==int
        if type(label) in [str,unicode]: label = ord(label)
        if type(olabel) in [str,unicode]: olabel = ord(olabel)
        self.label = label
        self.olabel = label if olabel is None else olabel
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        fst.AddArc(start,self.label,self.olabel,self.cost,end)

class COST:
    """An epsilon transition with the given cost."""
    def __init__(self,expr,cost):
        self.expr = asgen(expr)
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        state = fst.AddState()
        self.expr.generate(fst,start,state)
        fst.AddArc(state,epsilon,epsilon,self.cost,end)

class SEQ:
    """A sequence of FSTs.  Example: SEQ("AB","C","DEF")==STR("ABCDEF")."""
    def __init__(self,*args):
        self.args = [asgen(arg) for arg in args]
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        args = self.args
        for i in range(len(args)):
            next = fst.AddState() if i<len(args)-1 else end
            args[i].generate(fst,start,next)
            start = next

class Y:
    """A range of characters, as in Y("A-Za-z,.!0-9"), similar to regular expressions."""
    def __init__(self,s,cost=0.0,condition=lambda x:True):
        assert type(s)==str or type(s)==unicode
        self.s = s
        self.cost = cost
        self.condition = condition
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
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
                if self.condition(unichr(c)):
                    fst.AddArc(start,c,c,self.cost,end)

DIGITS = Y("0-9")
LETTER = Y("A-Za-z")
WS = Y(" \t\n")
ASCII = Y("@-~")

class RANGE:
    """A range of characters, as in RANGE("A","Z")."""
    def __init__(self,lo,hi,cost=0.0):
        assert type(lo)==str or type(lo)==unicode
        assert type(hi)==str or type(hi)==unicode
        self.lo = lo
        self.hi = hi
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        for c in range(ord(self.lo),ord(self.hi)+1):
            fst.AddArc(start,c,c,self.cost,end)

class ALT:
    """An set of alternatives.  ALT("A","B","C") is one of
    the three strings, written as A|B|C in regular expressions."""
    def __init__(self,*args):
        self.args = [asgen(arg) for arg in args]
        self.cost = 0.0
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc'),(start,end,fst)
        for arg in self.args:
            if type(arg)==str or type(arg)==unicode:
                arg = STR(arg,cost=self.cost)
            arg.generate(fst,start,end)

class OPT:
    """Zero or one repetitions.  OPT("A") is the analog
    of A? in regular expressions."""
    def __init__(self,expr):
        self.expr = asgen(expr)
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc')
        self.expr.generate(fst,start,end)
        fst.AddArc(start,epsilon,epsilon,0.0,end)

class STAR:
    """Zero or more repetitions.  STAR("A") is the analog
    of A* in regular expressions."""
    def __init__(self,expr,cost=0.0):
        self.expr = asgen(expr)
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc')
        start2 = fst.AddState()
        fst.AddArc(start,epsilon,epsilon,0.0,start2)
        fst.AddArc(start2,epsilon,epsilon,0.0,end)
        self.expr.generate(fst,start2,start2)

class PLUS:
    """One or more repetitions.  PLUS("A") is the analog
    of A+ in regular expressions."""
    def __init__(self,expr,cost=0.0):
        self.expr = asgen(expr)
        self.cost = cost
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc')
        start2 = fst.AddState()
        end2 = fst.AddState()
        fst.AddArc(start,epsilon,epsilon,0.0,start2)
        fst.AddArc(end2,epsilon,epsilon,0.0,end)
        self.expr.generate(fst,start2,end2)
        fst.AddArc(end2,epsilon,epsilon,0.0,start2)

class DICTIONARY:
    """Load a dictionary and add it to the FST.
    Equivalent to ALT(STR("line1"),STR("line2"),...)
    for the lines of the file.  If the file contains lines
    of the form "cost\ttext\n", then the cost is associated
    with the line, otherwise the cost is 0."""
    def __init__(self,fname):
        assert type(fname)==str and os.path.exists(fname),fname
        self.fname = fname
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc')
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
        assert type(fname)==str and os.path.exists(fname),fname
        raise Exception("unimplemented")
    def generate(self,fst,start,end):
        assert type(start)==int and type(end)==int and hasattr(fst,'AddArc')
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

if __name__=="__main__":
    fst = GEN(PLUS(ALT(PLUS("A"),PLUS("B"))))
    fst.Write("test.fst")

