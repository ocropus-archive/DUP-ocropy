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
    result = common.OcroFST()
    result.load(temp)
    os.unlink(temp)
    return result

def add_default_symtab(fst):
    """Adds a default symbol table to the given fst."""
    table = lig.SymbolTable(name="unicode")
    fst.SetInputSymbols(table)
    fst.SetOutputSymbols(table)
    return fst

class AlignerMixin:
    def getOcroFst(self):
        fst = self.getFst()
        return openfst2ocrofst(fst)
    def fstForLines(self,lines):
        self.startFst()
        for line in lines:
            self.addTranscription(line)
        return self.getFst()
    def ocroFstForLines(self,lines):
        self.startFst()
        for line in lines:
            self.addTranscription(line)
        return self.getOcroFst()
    def fstForFile(self,file):
        try:
            with codecs.open(file,"r","utf-8") as stream:
                return self.fstForLines(stream.readlines())
        except UnicodeDecodeError,e:
            raise Exception("bad unicode in "+file)
    def ocroFstForFile(self,file):
        try:
            with codecs.open(file,"r","utf-8") as stream:
                return self.ocroFstForLines(stream.readlines())
        except UnicodeDecodeError,e:
            raise Exception("bad unicode in "+file)

class DefaultAligner(AlignerMixin):
    def __init__(self):
        self.error="#"
        self.insert=""
        self.delete=""
        self.space_insert=5.0
        self.space_delete=5.0
        self.char_mismatch=8.0
        self.char_insert=2.0
        self.char_delete=10.0
        self.classifier_reject=None
        self.add_l2l=0.0
        self.add_c2l=0.0
        self.add_l2c=0.0
        self.exceptional_ligatures = []
        self.exceptional_cost = 0.0
        self.rewrites = [ ]
        self.lig = ligatures.lig
        self.sigout = True
        self.optimize = 0
    def startFst(self):
        self.fst = openfst.StdVectorFst()
    def getFst(self):
        fst = self.fst
        self.fst = None
        fst = optimize_openfst(fst,optimize=self.optimize)
        return fst
    def explodeTranscription(self,line):
        line = line.strip()
        line = re.sub(r'[ ~]+',' ',line)
        return explode_transcription(line)
    def addTranscription(self,line):
        line = self.explodeTranscription(line)
        self.addCodes(line)
    def addCodes(self,line,accept=0.0):
        fst = self.fst
        lig = self.lig
        state = fst.Start()
        if state<0:
            state = fst.AddState()
            fst.SetStart(state)
        states = [state]
        for i in range(len(line)):
            states.append(fst.AddState())
        for i in range(len(line)):
            s = line[i]
            c = lig.ord(s)
            start = states[i]
            next = states[i+1]

            # insert characters or ligatures as single tokens

            if len(s)==1 or self.add_l2l is not None:
                assert c is not None,"ligature [%s] not found in ligature table"%s
                cost = 0.0 if len(s)==1 else self.add_l2l
                fst.AddArc(start,c,c,cost,next)
                if self.classifier_reject is not None:
                    fst.AddArc(start,lig.ord("~"),c,self.classifier_reject,next)

            # insert rewrites

            if self.rewrites is not None:
                for s,gt,c in self.rewrites:
                    add_between(fst,start,next,s,gt,c,lig=lig)
                    # FIXME replace the rest of the loops below with add_between as well

            # allow insertion of spaces with some cost

            if self.space_insert is not None:
                fst.AddArc(next,epsilon,space,self.space_insert,next)

            # space is special (since we use separate skip/insertion self)

            if s==" ":
                # space transition
                fst.AddArc(start,space,space,0.0,next)
                # space skip transition
                if self.space_delete is not None:
                    fst.AddArc(start,epsilon,space,self.space_delete,next)
                continue

            # allow insertion of a character relative to ground truth

            if self.char_insert is not None:
                fst.AddArc(start,sigma,lig.ord(self.insert),self.char_insert,start)

            # allow character mismatches

            if self.char_mismatch is not None:
                if self.sigout:
                    fst.AddArc(start,sigma,lig.ord(s),self.char_mismatch,next)
                else:
                    fst.AddArc(start,sigma,lig.ord(self.error),self.char_mismatch,next)

            # allow deletion of a character relative to ground truth

            if self.char_delete is not None:
                fst.AddArc(start,epsilon,lig.ord(self.delete),self.char_delete,next)

            # insert character-to-ligature

            if len(s)>1 and self.add_c2l is not None:
                add_between(fst,start,next,list(s),[s],self.add_c2l,lig=lig)

            # insert ligature-to-characters

            if self.add_l2c is not None:
                candidate = "".join(line[i:i+4])
                for s in ligatures.common_ligatures(candidate):
                    cc = lig.ord(s)
                    nnext = states[i+len(s)]
                    fst.AddArc(start,cc,cc,self.add_l2c,nnext)

            if self.exceptional_cost is not None:
                candidate = "".join(line[i:i+4])
                for s in self.exceptional_ligatures:
                    if candidate.startswith(s):
                        cc = lig.ord(s)
                        nnext = states[i+len(s)]
                        fst.AddArc(start,lig.ord("~"),cc,self.add_l2c,nnext)

        # also allow junk at the end

        if self.char_insert is not None:
            fst.AddArc(states[-1],sigma,lig.ord(self.insert),self.char_insert,states[-1])

        fst.SetFinal(states[-1],accept)
