import sys,os,re
import openfst
import ocropus,iulib
import common

space = ord(" ")
epsilon = openfst.epsilon
assert epsilon==ocropus.L_EPSILON
rho = ocropus.L_RHO
sigma = ocropus.L_SIGMA
phi = ocropus.L_PHI

import ligatures

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

simple_costs = Costs(
                      # insertions of spaces relative to ground truth
                      space_insert=None,
                      # deletions of spaces relative to ground truth
                      space_delete=None,
                      # mismatches of characters relative to ground truth
                      char_mismatch=None,
                      # insertions of characters relative to ground truth
                      char_insert=None,
                      # deletions of characters relative to ground truth
                      char_delete=None,
                      # allow multiple character in transcript to be represented by one thing in lattice
                      # if they are known to be common ligatures
                      ligatures2=None,
                      ligatures3=None,
                      # allow multiple character in transcript to be represented by one thing in lattice
                      misseg2=None,
                      misseg3=None,
                      # add rejects as sigmas with a given cost
                      reject=1.0,
                      # ligatures in both lattice and transcript
                      add_l2l=None,
                      # ligatures as characters in both lattice and transcript
                      add_c2c=None,
                      # ligatures in lattice, characters in transcript                      
                      add_l2c=None,
                      # characters in lattice, ligatures in transcript
                      add_c2l=None,
                      )

def make_costs(err=5.0,space_err=5.0,ligature=0.0,ligchange=1.0,reject=0.1):
    return Costs(
        # insertions of spaces relative to ground truth
        space_insert=space_err,
        # deletions of spaces relative to ground truth
        space_delete=space_err,
        # mismatches of characters relative to ground truth
        char_mismatch=err,
        # insertions of characters relative to ground truth
        char_insert=err,
        # deletions of characters relative to ground truth
        char_delete=err,
        # allow multiple character in transcript to be represented by one thing in lattice
        # if they are known to be common ligatures
        ligatures2=ligature,
        ligatures3=ligature,
        # allow multiple character in transcript to be represented by one thing in lattice
        misseg2=None,
        misseg3=None,
        # add rejects as sigmas with a given cost
        reject=reject,
        # ligatures in both lattice and transcript
        add_l2l=ligature,
        # characters in lattice, ligatures in transcript
        add_c2l=ligature,
        # ligatures as characters in both lattice and transcript
        add_c2c=ligchange,
        # ligatures in lattice, characters in transcript                      
        add_l2c=ligchange,
        )

default_costs = make_costs()

def add_line_to_fst(fst,line,
                costs=default_costs,accept=0.0,
                lig=ligatures.lig):
    """Add a line (given as a list of strings) to an fst."""
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
            # FIXME -- There are several things wrong here: sigma isn't working,
            # and more importantly, even if it were, we'd like to be able to 
            # override (ignore) the cost on the input value here
            fst.AddArc(start,lig.ord("_"),lig.ord("~"),costs.reject,next)
            fst.AddArc(start,lig.ord("~"),lig.ord("~"),costs.reject,next)
            fst.AddArc(start,sigma,lig.ord("~"),costs.reject,next)
            for i in range(32,128): fst.AddArc(start,i,lig.ord("~"),costs.reject,next)
            fst.AddArc(start,epsilon,lig.ord("~"),costs.reject,next)
            continue

        # allow insertion of a character relative to ground truth
        
        if costs.char_insert is not None:
            fst.AddArc(start,sigma,epsilon,costs.char_insert,start)

        # allow character mismatches

        if costs.char_mismatch is not None:
            fst.AddArc(start,sigma,c,costs.char_mismatch,next)

        # allow deletion of a character relative to ground truth

        if costs.char_delete is not None:
            fst.AddArc(start,epsilon,c,costs.char_insert,next)

        # insert characters or ligatures as single tokens

        if len(s)==1 or costs.add_l2l>=0:
            assert c is not None,"ligature [%s] not found in ligature table"%s
            cost = 0.0 if len(s)==1 else costs.add_l2l
            fst.AddArc(start,c,c,cost,next)

        # insert ligatures as exploded list of characters with some cost

        if len(s)>1 and costs.add_c2c>=0.0:
            state = start
            for j in range(len(s)):
                cc = lig.ord(s[j])
                nstate = next if j==len(s)-1 else fst.AddState()
                cost = costs.add_c2c if j==0 else 0.0
                fst.AddArc(state,cc,cc,cost,nstate)
                state = nstate

        # insert ligature-to-character

        if len(s)>1 and costs.add_l2c>=0.0:
            state = start
            for j in range(len(s)):
                cc = lig.ord(s[j])
                nstate = next if j==len(s)-1 else fst.AddState()
                cost = costs.add_l2c if j==0 else 0.0
                fst.AddArc(state,c if j==0 else epsilon,cc,cost,nstate)
                state = nstate

        # insert character-to-ligature

        if len(s)>1 and costs.add_c2l>=0.0:
            state = start
            for j in range(len(s)):
                cc = lig.ord(s[j])
                nstate = next if j==len(s)-1 else fst.AddState()
                cost = costs.add_c2l if j==0.0 else 0.0
                fst.AddArc(state,cc,c if j==0 else epsilon,cost,nstate)
                state = nstate

        # allow segmentation error with two characters

        if i<len(line)-2 and " " not in line[i:i+2] and costs.misseg2 is not None:
            state1 = fst.AddState()
            fst.AddArc(start,lig.ord("~"),sigma,costs.misseg2,state1)
            fst.AddArc(state1,epsilon,sigma,0.0,states[i+2])

        # allow segmentation error with three characters in ground truth

        if i<len(line)-3 and " " not in line[i:i+3] and costs.misseg3 is not None:
            state1,state2 = (fst.AddState(),fst.AddState())
            fst.AddArc(start,lig.ord("~"),sigma,costs.misseg3,state1)
            fst.AddArc(state1,epsilon,sigma,0.0,state2)
            fst.AddArc(state2,epsilon,sigma,0.0,states[i+3])

    fst.SetFinal(states[-1],accept)

def make_line_openfst(lines,lig=ligatures.lig,optimize=0):
    """Given a list of text lines, construct a corresponding FST.
    Each text line is a list of strings."""
    assert type(lines)==list
    fst = Fst()
    count = 0
    for line in lines:
        if type(line) in [str,unicode]:
            line = explode_transcription(line)
        count += 1
        add_line_to_fst(fst,line,lig=lig)
    if not optimize:
        det = fst
    elif optimize==1:
        fst.Write("_l_.fst")
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
    table = lig.SymbolTable(name="unicode")
    fst.SetInputSymbols(table)
    fst.SetOutputSymbols(table)
    fst.Write("_temp.fst")
    return det

def make_line_fst(lines):
    openfst = make_line_openfst(lines)
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
    
def load_transcription(file,use_ligatures=0):
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
            lines = [re.sub(r'_','~',l) for l in lines]
        return make_line_fst(lines)

def make_alignment_fst(transcriptions):
    """Takes a string or a list of strings that are transcriptions and
    constructs an FST suitable for alignment."""
    if isinstance(transcriptions,ocrolib.OcroFST):
        return transcriptions
    if type(transcriptions) in [str,unicode]:
        transcriptions = [transcriptions]
    transcriptions = [explode_transcription(s) for s in transcriptions]
    return transcriptions
    
def simple_line_fst(line,lig=ligatures.lig):
    """Add a line (given as a list of strings) to an fst."""
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

