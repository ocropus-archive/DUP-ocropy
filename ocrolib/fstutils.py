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

def implode_transcription(transcription):
    """Takes a list of characters or strings and implodes them into
    a transcription."""
    def quote(s):
        assert len(s)<=4,"ligatures can consist of at most 4 characters"
        s = re.sub(r'_','~',s)
        if len(s)>1: return "_"+s+"_"
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

