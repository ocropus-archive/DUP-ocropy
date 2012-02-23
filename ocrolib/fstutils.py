# -*- encoding: utf-8 -*-

################################################################
### Utilities for constructing various FSTs.  
### In particular, this contains code for constructing
### FSTs that align ground truth with OCR output.
################################################################

import sys,os,re,codecs
import openfst
import iulib
#import ocropus
import common
import ligatures
import ocrofst
from collections import namedtuple
import docproc

class Record:
    def __init__(self,**kw):
        self.__dict__.update(kw)
    def like(self,obj):
        self.__dict__.update(obj.__dict__)
        return self

epsilon = openfst.epsilon
sigma = -3 # FIXME
space = ligatures.lig.ord(" ")
reject = ligatures.lig.ord("~")

Fst = openfst.StdVectorFst

class BadTranscriptionSyntax:
    pass

def check_transcription(transcription):
    """Checks the syntax of transcriptions.  Transcriptions may have ligatures
    enclosed like "a_ffi_ne", but the text between the ligatures may not be
    longer than 4 characters (to catch typos)."""
    groups = re.split(r'(_.*?_)',transcription)
    for i in range(1,len(groups),2):
        if len(groups[i])>6: raise BadTranscriptionSyntax()
    return

def explode_transcription(transcription):
    """Explode a transcription into a list of strings.  Characters are usually
    exploded into individual list elements, unless they are enclosed as in
    "a_ffi_ne", in which case the string between the "_" is considered a ligature.
    Backslash may be used to escape special characters, including "\" and "_"."""
    if type(transcription)==list:
        return transcription
    elif type(transcription) in [str,unicode]:
        check_transcription(transcription)
        transcription = re.sub("[\000-\037]","~",transcription)
        transcription = transcription.replace("\\_","\0x4").replace("\\\\","\0x3")
        groups = re.split(r'(_.{1,4}?_)',transcription)
        for i in range(1,len(groups),2):
            groups[i] = groups[i].strip("_")
        result = []
        for i in range(len(groups)):
            if i%2==0:
                s = groups[i]
                for j in range(len(s)):
                    result.append(s[j])
            else:
                result.append(groups[i])
        result = [s.replace("\0x4","_").replace("\0x3","\\") for s in result]
        return result
    raise Exception("bad transcription type")

def implode_transcription(transcription,maxlig=4):
    """Takes a list of characters or strings and implodes them into
    a transcription."""
    def quote(s):
        assert len(s)<=maxlig,"ligatures can consist of at most 4 characters"
        if "~" in s: s = "~"
        s = re.sub(r'[\0]+','',s)
        s = re.sub(r' +',' ',s)
        l = len(s)
        s = re.sub(r'_',"\\_",s)
        s = re.sub(r"\\\\","\\\\\\\\",s)
        if l>1: return "_"+s+"_"
        else: return s
    return "".join([quote(s) for s in transcription])

def optimize_openfst(fst,optimize=1):
    """Returns a minimized version of the input fst.  The result
    may or may not be identical to the input."""
    if optimize==0:
        return fst
    elif optimize==1:
        det = Fst()
        mapper = openfst.StdEncodeMapper(openfst.kEncodeLabels,openfst.ENCODE)
        openfst.Encode(fst,mapper)
        openfst.RmEpsilon(fst)
        openfst.Determinize(fst,det)
        openfst.Minimize(det)
        fst = det
        openfst.Decode(fst,mapper)
    elif optimize==2:
        det = Fst()
        openfst.RmEpsilon(fst)
        openfst.Determinize(fst,det)
        openfst.Minimize(det)
        fst = det
    return fst

def openfst2ocrofst(openfst):
    """Convert an OpenFST transducer to an OcroFST transducer."""
    temp = "/tmp/%d.fst"%os.getpid()
    openfst.Write(temp)
    result = ocrofst.OcroFST()
    result.load(temp)
    os.unlink(temp)
    return result

def add_default_symtab(fst):
    """Adds a default symbol table to the given fst."""
    table = lig.SymbolTable(name="unicode")
    fst.SetInputSymbols(table)
    fst.SetOutputSymbols(table)
    return fst

def add_between(fst,frm,to,l1,l2,cost,lig=ligatures.lig):
    assert type(l1)==list
    assert type(l2)==list
    state = frm
    n = max(len(l1),len(l2))
    for i in range(n):
        a = l1[i] if i<len(l1) else 0
        if type(a) in [str,unicode]: a = lig.ord(a)
        b = l2[i] if i<len(l2) else 0
        if type(b) in [str,unicode]: b = lig.ord(b)
        c = cost if i==0 else 0.0
        next = to if i==n-1 else fst.AddState()
        # print [(x,type(x)) for x in state,a,b,c,next]
        fst.AddArc(state,a,b,c,next)
        state = next

import codecs
import openfst

from ligatures import lig


def alignment_fst(line,maxtrim=4,trimcost=2.0,edcost=4,rcost=4,lcost2=3,lcost3=4,confusions=[],ltable=lig):
    fst = ocrofst.OcroFST()
    state = fst.AddState()
    fst.SetStart(state)
    states = [state]
    for i in range(len(line)):
        states.append(fst.AddState())
    ntrim = min(len(line)/4,maxtrim)
    for i in range(ntrim):
        fst.AddArc(states[i],epsilon,ord("~"),trimcost,states[i+1])
    for i in range(len(line)-ntrim,len(line)):
        fst.AddArc(states[i],epsilon,ord("~"),trimcost,states[i+1])
    for i in range(len(line)):
        s = line[i]
        c = ord(s)
        start = states[i]
        next = states[i+1]

        # space is special (since we use separate skip/insertion self)

        # insertion of space
        fst.AddArc(next,space,space,0.0,next)
        # insertion of other character
        fst.AddArc(start,sigma,ord("~"),edcost,start)

        if s==" ":
            # space transition
            fst.AddArc(start,space,space,0.0,next)
            # skip space
            fst.AddArc(start,epsilon,space,0.0,next)
            continue

        if s in ["~","_"]:
            # common ground-truth indicators of errors, unrecognizable characters
            fst.AddArc(start,sigma,ord("~"),4.0,start) # also allow repetition with some cost
            fst.AddArc(start,sigma,ord("~"),0.0,next)
            continue

        # add character transition
        fst.AddArc(start,c,c,0.0,next)
        # mismatch between input and transcription
        fst.AddArc(start,sigma,ord("_"),rcost,next)
        # deletion in lattice
        fst.AddArc(start,epsilon,ord("~"),edcost,next)
        # insertion in lattice
        fst.AddArc(start,sigma,epsilon,edcost,next)

    # explicit transition for confusions; note that multi-character
    # outputs are always treated as ligatures

    for i in range(0,len(line)):
        for u,v,cost in confusions:
            if i+len(v)>len(line): continue
            if line[i:i+len(v)]==v:
                start = states[i]
                end = states[i+len(v)]
                for j in range(len(u)-1):
                    next = fst.AddState()
                    fst.AddArc(start,ord(u[j]),epsilon,0.0,next)
                    start = next
                fst.AddArc(start,ord(u[-1]),ltable.ord(v),cost,end)

    # explicit transitions for ligatures: use epsilon on input, use encoded ligatures on output,

    inf = 1e33
    if lcost2<inf:
        for i in range(0,len(line)-2):
            s = line[i:i+2]
            if " " in s: continue
            start = states[i]
            catch = fst.AddState()
            fst.AddArc(start,ord("~"),ltable.ord(s),0.0,catch)
            fst.AddArc(start,sigma,ltable.ord(s),lcost2,catch)
            fst.AddArc(catch,ord(line[i+2]),ord(line[i+2]),0.0,states[i+3])

    if lcost3<inf:
        for i in range(0,len(line)-3):
            s = line[i:i+3]
            if " " in s: continue
            start = states[i]
            catch = fst.AddState()
            fst.AddArc(start,ord("~"),ltable.ord(s),0.0,next)
            fst.AddArc(start,sigma,ltable.ord(s),lcost3,next)
            fst.AddArc(catch,ord(line[i+3]),ord(line[i+3]),0.0,states[i+4])

    # insertions at beginning or end
    fst.AddArc(states[0],sigma,epsilon,edcost,states[0])
    fst.AddArc(states[-1],sigma,epsilon,edcost,states[-1])
    # space insertions at beginning or end
    fst.AddArc(states[0],space,epsilon,0.0,states[0])
    fst.AddArc(states[-1],space,epsilon,0.0,states[-1])
    # set final state
    fst.SetFinal(states[-1],0.0)
    # print line; fst.save("_aligner.fst"); sys.exit(0)
    return fst

def load_text_file_as_fst(fname):
    with open(fname) as stream:
        text = stream.read()
    text = re.sub('[ \r\n\t]+',' ',text)
    text = text.strip()
    return alignment_fst(text)

from pylab import *

def compute_alignment(fst,rseg,lmodel,beam=1000,ltable=lig):
    result = ocrofst.beam_search(fst,lmodel,100)
    v1,v2,ins,outs,costs = result
    sresult = []
    scosts = []
    segs = []
    n = len(ins)
    i = 1
    while i<n:
        if outs[i]==ord(" "):
            sresult.append(" ")
            scosts.append(costs[i])
            segs.append((0,0))
            i += 1
            continue
        # pick up ligatures indicated by the recognizer (multiple sequential 
        # output characters with the same input segments)
        j = i+1
        while j<n and ins[j]==ins[i]:
            j += 1
        cls = "".join([ltable.chr(c) for c in outs[i:j]])
        sresult.append(cls)
        scosts.append(sum(costs[i:j]))
        start = (ins[i]>>16)
        end = (ins[i]&0xffff)
        segs.append((start,end))
        i = j

    assert len(sresult)==len(segs)
    assert len(scosts)==len(segs)

    rseg_boxes = docproc.seg_boxes(rseg)

    bboxes = []

    rmap = zeros(amax(rseg)+1,'i')
    for i in range(1,len(segs)):
        start,end = segs[i]
        if start==0 or end==0: continue
        rmap[start:end+1] = i
        bboxes.append(common.rect_union(rseg_boxes[start:end+1]))
    assert rmap[0]==0

    cseg = zeros(rseg.shape,'i')
    for i in range(cseg.shape[0]):
        for j in range(cseg.shape[1]):
            cseg[i,j] = rmap[rseg[i,j]]

    assert len(segs)==len(sresult) 
    assert len(segs)==len(scosts)

    assert amin(cseg)==0,"amin(cseg)!=0 (%d,%d)"%(amin(cseg),amax(cseg))

    return Record(output_l=sresult,
                  cseg = cseg,
                  costs = scosts)

def simple_line_fst(line,lig=ligatures.lig):
    """Add a line (given as a list of strings) to an fst."""
    fst = ocrofst.OcroFST()
    state = fst.newState()
    fst.setStart(state)
    states = [state]
    for i in range(len(line)):
        states.append(fst.newState())
    for i in range(len(line)):
        s = line[i]
        c = lig.ord(s)
        start = states[i]
        next = states[i+1]
        fst.addTransition(start,next,c)
    fst.setAccept(states[-1],0.0)
    return fst
