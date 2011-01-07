import sys,os,re
import openfst
import ocropus,iulib
import common

Fst = openfst.StdVectorFst

def add_line(fst,s):
    s = re.sub('\s+',' ',s)
    state = fst.Start()
    if state<0:
        state = fst.AddState()
        fst.SetStart(state)
    for i in range(len(s)):
        c = ord(s[i])
        if c>=32 and c<128:
            pass
        else:
            # print "skipping %d"%c
            raise Exception("bad character %d in '%s' (%s)"%(c,s,type(s)))
            continue
        nstate = fst.AddState()
        if s[i]==" ":
            fst.AddArc(state,c,c,0.0,nstate)
            fst.AddArc(state,openfst.epsilon,openfst.epsilon,1.0,nstate)
            fst.AddArc(nstate,c,c,0.0,nstate)
        else:
            fst.AddArc(state,c,c,0.0,nstate)
        state = nstate
    fst.SetFinal(state,0.0001)

def make_line_fst(lines):
    fst = Fst()
    count = 0
    for line in lines:
        count += 1
        try:
            add_line(fst,line)
        except:
            print "on line",count
            raise
    det = Fst()
    openfst.Determinize(fst,det)
    openfst.Minimize(det)
    temp = "/tmp/%d.fst"%os.getpid()
    det.Write(temp)
    result = common.OcroFST()
    result.load(temp)
    os.unlink(temp)
    return result

def lines_of_file(file):
    for line in open(file).readlines():
        if line[-1]=="\n": line = line[:-1]
        if line=="": continue
        yield unicode(line,"utf-8")

def load_text_file_as_fst(file):
    lines = list(lines_of_file(file))
    return make_line_fst(lines)
