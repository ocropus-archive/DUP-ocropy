# -*- encoding: utf-8 -*-

import sys,os,re
import openfst
import ocropus,iulib
import common

space = ord(" ")
epsilon = openfst.epsilon
assert epsilon==ocropus.L_EPSILON
rho = ocropus.L_RHO
sigma = ocropus.L_RHO
phi = ocropus.L_PHI

import ligatures

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

Fst = openfst.StdVectorFst

class Costs:
    def __init__(self,**kw):
        self.__dict__.update(kw)

default_costs = {}

default_costs["strict"] = Costs(
        # output used to indicate that something is wrong
        error="#",
        insert="",
        delete="",
        # insertions of spaces relative to ground truth
        space_insert=None,
        # deletions of spaces relative to ground truth
        space_delete=None,
        # mismatches of characters relative to ground truth
        char_mismatch=50.0,
        # insertions of characters relative to ground truth
        char_insert=None,
        # deletions of characters relative to ground truth
        char_delete=None,
        # add ground truth rejects as sigmas with a given cost
        reject=None,
        # add classifier rejects as sigmas
        classifier_reject=None,
        # ligatures in both lattice and transcript
        add_l2l=0.0,
        # characters in lattice, ligatures in transcript
        add_c2l=None,
        # ligatures in lattice, ligature-characters in transcript                      
        add_l2c=0.0,
        # these ligatures are allowed to match junk characters
        # in transcript; this is useful for getting models for new
        # ligatures
        exceptional_ligatures = [],
        exceptional_cost = 0.0,
        # list of rewrites
        rewrites = [ ],
        # output transcription rather than error
        sigout = False,
        )

default_costs["default"] = Costs(
        # output used to indicate that something is wrong
        error="#",
        insert="",
        delete="",
        # insertions of spaces relative to ground truth
        space_insert=7.0,
        # deletions of spaces relative to ground truth
        space_delete=7.0,
        # mismatches of characters relative to ground truth
        char_mismatch=10.0,
        # insertions of characters relative to ground truth
        char_insert=7.0,
        # deletions of characters relative to ground truth
        char_delete=7.0,
        # add ground truth rejects as sigmas with a given cost
        reject=None,
        # add classifier rejects as sigmas
        classifier_reject=None,
        # ligatures in both lattice and transcript
        add_l2l=0.0,
        # characters in lattice, ligatures in transcript
        add_c2l=0.0,
        # ligatures in lattice, ligature-characters in transcript                      
        add_l2c=0.0,
        # these ligatures are allowed to match junk characters
        # in transcript; this is useful for getting models for new
        # ligatures
        exceptional_ligatures = [],
        exceptional_cost = 0.0,
        # list of rewrites
        rewrites = [ ],
        # output transcription rather than error
        sigout = True,
        )

default_costs["flexible"] = Costs(
        # output used to indicate that something is wrong
        error="#",
        insert="",
        delete="",
        # insertions of spaces relative to ground truth
        space_insert=3.0,
        # deletions of spaces relative to ground truth
        space_delete=3.0,
        # mismatches of characters relative to ground truth
        char_mismatch=10.0,
        # insertions of characters relative to ground truth
        char_insert=3.0,
        # deletions of characters relative to ground truth
        char_delete=3.0,
        # add ground truth rejects as sigmas with a given cost
        reject=None,
        # add classifier rejects as sigmas
        classifier_reject=None,
        # ligatures in both lattice and transcript
        add_l2l=0.0,
        # characters in lattice, ligatures in transcript
        add_c2l=0.0,
        # ligatures in lattice, ligature-characters in transcript                      
        add_l2c=0.0,
        # these ligatures are allowed to match junk characters
        # in transcript; this is useful for getting models for new
        # ligatures
        exceptional_ligatures = [],
        exceptional_cost = 0.0,
        # list of rewrites
        rewrites = [ ],
        # output transcription rather than error
        sigout = True,
        )

default_costs["lenient"] = Costs(
        # output used to indicate that something is wrong
        error="#",
        insert="",
        delete="",
        # insertions of spaces relative to ground truth
        space_insert=5.0,
        # deletions of spaces relative to ground truth
        space_delete=5.0,
        # mismatches of characters relative to ground truth
        char_mismatch=5.0,
        # insertions of characters relative to ground truth
        char_insert=5.0,
        # deletions of characters relative to ground truth
        char_delete=5.0,
        # add ground truth rejects as sigmas with a given cost
        reject=None,
        # add classifier rejects as sigmas
        classifier_reject=None,
        # ligatures in both lattice and transcript
        add_l2l=0.0,
        # characters in lattice, ligatures in transcript
        add_c2l=0.0,
        # ligatures in lattice, ligature-characters in transcript                      
        add_l2c=0.0,
        # these ligatures are allowed to match junk characters
        # in transcript; this is useful for getting models for new
        # ligatures
        exceptional_ligatures = [],
        exceptional_cost = 0.0,
        # list of rewrites
        rewrites = [ ],
        # output transcription rather than error
        sigout = True,
        )

def add_between(fst,frm,to,l1,l2,cost,lig=ligatures.lig):
    if type(l1)!=list: l1 = explode_transcription(l1)
    if type(l2)!=list: l2 = explode_transcription(l2)
    state = frm
    n = max(len(l1),len(l2))
    for i in range(n):
        a = lig.ord(l1[i]) if i<len(l1) else 0
        b = lig.ord(l2[i]) if i<len(l2) else 0
        c = cost if i==0 else 0.0
        next = to if i==n-1 else fst.AddState()
        # print [(x,type(x)) for x in state,a,b,c,next]
        fst.AddArc(state,a,b,c,next)
        state = next

def add_line_to_fst(fst,line,
                    costs="default",
                    accept=0.0,
                    lig=ligatures.lig):
    """Add a line (given as a list of strings) to an fst."""
    state = fst.Start()
    if costs is None:
        costs = "default"
    if type(costs)==str: 
        costs = default_costs[costs]
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

        if len(s)==1 or costs.add_l2l is not None:
            assert c is not None,"ligature [%s] not found in ligature table"%s
            cost = 0.0 if len(s)==1 else costs.add_l2l
            fst.AddArc(start,c,c,cost,next)
            if costs.classifier_reject is not None:
                fst.AddArc(start,lig.ord("~"),c,costs.classifier_reject,next)

        # insert rewrites

        if costs.rewrites is not None:
            for s,gt,c in costs.rewrites:
                add_between(fst,start,next,s,gt,c,lig=lig)
                # FIXME replace the rest of the loops below with add_between as well

        # allow insertion of spaces with some cost
                
        if costs.space_insert is not None:
            fst.AddArc(next,epsilon,space,costs.space_insert,next)

        # space is special (since we use separate skip/insertion costs)

        if s==" ":
            # space transition
            fst.AddArc(start,space,space,0.0,next)
            # space skip transition
            if costs.space_delete is not None:
                fst.AddArc(start,epsilon,space,costs.space_delete,next)
            continue

        if s=="~" and costs.reject is not None:
            fst.AddArc(start,lig.ord("~"),lig.ord("~"),costs.reject,next)
            fst.AddArc(start,sigma,lig.ord("~"),costs.reject,next)
            fst.AddArc(start,epsilon,lig.ord("~"),costs.reject,next)
            continue

        # allow insertion of a character relative to ground truth
        
        if costs.char_insert is not None:
            fst.AddArc(start,sigma,lig.ord(costs.insert),costs.char_insert,start)

        # allow character mismatches

        if costs.char_mismatch is not None:
            if costs.sigout:
                fst.AddArc(start,sigma,lig.ord(s),costs.char_mismatch,next)
            else:
                fst.AddArc(start,sigma,lig.ord(costs.error),costs.char_mismatch,next)

        # allow deletion of a character relative to ground truth

        if costs.char_delete is not None:
            fst.AddArc(start,epsilon,lig.ord(costs.delete),costs.char_delete,next)

        # insert character-to-ligature

        if len(s)>1 and costs.add_c2l is not None:
            add_between(fst,start,next,list(s),[s],costs.add_c2l,lig=lig)

        # insert ligature-to-characters

        if costs.add_l2c is not None:
            candidate = "".join(line[i:i+4])
            for s in ligatures.common_ligatures(candidate):
                cc = lig.ord(s)
                nnext = states[i+len(s)]
                fst.AddArc(start,cc,cc,costs.add_l2c,nnext)

        if costs.exceptional_cost is not None:
            candidate = "".join(line[i:i+4])
            for s in costs.exceptional_ligatures:
                if candidate.startswith(s):
                    cc = lig.ord(s)
                    nnext = states[i+len(s)]
                    fst.AddArc(start,lig.ord("~"),cc,costs.add_l2c,nnext)

    fst.SetFinal(states[-1],accept)

def make_line_openfst(lines,lig=ligatures.lig,optimize=0,symtab="default",costs=None):
    """Given a list of text lines, construct a corresponding FST.
    Each text line is a list of strings."""
    assert type(lines)==list
    fst = Fst()
    count = 0
    for line in lines:
        if type(line) in [str,unicode]:
            line = explode_transcription(line)
        count += 1
        add_line_to_fst(fst,line,lig=lig,costs=costs)
    if not optimize:
        det = fst
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
    if symtab=="default":
        table = lig.SymbolTable(name="unicode")
    elif symtab is not None:
        table = symtab
    if symtab is not None:
        fst.SetInputSymbols(table)
        fst.SetOutputSymbols(table)
    fst.Write("_temp.fst")
    return det

def make_line_fst(lines,costs=None):
    openfst = make_line_openfst(lines,costs=costs)
    temp = "/tmp/%d.fst"%os.getpid()
    openfst.Write(temp)
    result = common.OcroFST()
    result.load(temp)
    os.unlink(temp)
    return result

def lines_of_file(file):
    """Returns the lines contained in a textfile as a list
    of unicode strings."""
    for line in open(file).readlines():
        if line[-1]=="\n": line = line[:-1]
        if line=="": continue
        line = line.strip()
        line = re.sub(r'[ 	]+',' ',line)
        try:
            u = unicode(line,"utf-8")
        except:
            raise Exception("bad unicode in transcription: %s: %s"%(file,line))
        yield u

def load_text_file_as_fst(file):
    """Use load_transcription instead."""
    return load_transcription(file)
    
def load_transcription(file,use_ligatures=1,costs=None):
    """Load a text file as a transcription.  This handles
    notation like "a_ffi_ne" for ligatures, "\" escapes, and
    and "~" for reject characters."""
    if file[-4:]==".fst":
        fst = common.OcroFST()
        fst.load(file)
        return fst
    else:
        lines = list(lines_of_file(file))
        if not use_ligatures:
            lines = [re.sub(r'_','',l) for l in lines]
        lines = [line.strip() for line in lines]
        return make_line_fst(lines,costs=costs)

def make_alignment_fst(transcriptions,use_ligatures=1):
    """Takes a string or a list of strings that are transcriptions and
    constructs an FST suitable for alignment."""
    if isinstance(transcriptions,ocrolib.OcroFST):
        return transcriptions
    if type(transcriptions) in [str,unicode]:
        transcriptions = [transcriptions]
    if use_ligatures:
        transcriptions = [explode_transcription(s) for s in transcriptions]
    else:
        transcriptions = [list(s) for s in transcriptions]
    fst = make_line_fst(transcriptions)
    return fst
    
def simple_line_fst(line,lig=ligatures.lig):
    """Add a line (given as a list of strings) to an fst."""
    line = line_cleanup(line)
    fst = common.OcroFST()
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

