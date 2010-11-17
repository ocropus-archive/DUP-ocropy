import sys,os,re,glob,math,glob,signal
import iulib,ocropus,narray
from pylab import *
from utils import *

def rseg_map(inputs):
    """This takes the FST produces by a beam search and looks at
    the input labels.  The input labels contain the correspondence
    between the rseg label and the character.  These are put into
    a dictionary and returned.  This is used for alignment between
    a segmentation and text."""
    n = inputs.length()
    segs = []
    for i in range(n):
        start = inputs.at(i)>>16
        end = inputs.at(i)&0xffff
        segs.append((start,end))
    n = amax([s[1] for s in segs])+1
    count = 0
    map = zeros(n,'i')
    for i in range(len(segs)):
        start,end = segs[i]
        if start==0 or end==0: continue
        count += 1
        for j in range(start,end+1):
            map[j] = count
    return map

def recognize_and_align(image,linerec,lmodel,beam=1000,nocseg=0):
    """Perform line recognition with the given line recognizer and
    language model.  Outputs an object containing the result (as a
    Python string), the costs, the rseg, the cseg, the lattice and the
    total cost.  The recognition lattice needs to have rseg's segment
    numbers as inputs (pairs of 16 bit numbers); SimpleGrouper
    produces such lattices.  cseg==None means that the connected
    component renumbering failed for some reason."""

    # run the recognizer
    lattice = ocropus.make_OcroFST()
    rseg = iulib.intarray()
    linerec.recognizeLineSeg(lattice,rseg,image)

    # perform the beam search through the lattice and the model
    v1 = iulib.intarray()
    v2 = iulib.intarray()
    ins = iulib.intarray()
    outs = iulib.intarray()
    costs = iulib.floatarray()
    ocropus.beam_search(v1,v2,ins,outs,costs,lattice,lmodel,beam)

    # do the conversions
    print "OUTS",[outs.at(i) for i in range(outs.length())]
    result = intarray_as_string(outs,skip0=0)
    print "RSLT",result

    # compute the cseg
    if not nocseg:
        rmap = rseg_map(ins)
        cseg = None
        if len(rmap)>1:
            cseg = iulib.intarray()
            cseg.copy(rseg)
            try:
                for i in range(cseg.length()):
                    cseg.put1d(i,int(rmap[rseg.at1d(i)]))
            except IndexError:
                raise Exception("renumbering failed")
    else:
        cseg = None

    # return everything we computed
    return Record(image=image,
                  output=result,
                  raw=outs,
                  costs=costs,
                  rseg=rseg,
                  cseg=cseg,
                  lattice=lattice,
                  cost=iulib.sum(costs))

def as_intarray(a):
    return array([a.at(i) for i in range(a.length())])

def compute_alignment_old(lattice,rseg,lmodel,beam=10000):
    """Given a lattice produced by a recognizer, a raw segmentation,
    and a language model, computes the best solution, the cseg, and
    the corresponding costs.  These are returned as Python data structures.
    The recognition lattice needs to have rseg's segment numbers as inputs
    (pairs of 16 bit numbers); SimpleGrouper produces such lattices."""

    # perform the beam search through the lattice and the model
    v1 = narray.intarray()
    v2 = narray.intarray()
    ins = narray.intarray()
    outs = narray.intarray()
    costs = narray.floatarray()
    ocropus.beam_search(v1,v2,ins,outs,costs,lattice,lmodel,beam)

    # do the conversions
    result = intarray_as_string(outs)

    # compute the cseg
    rmap = rseg_map(ins)
    cseg = None
    if len(rmap)>1:
        cseg = narray.intarray()
        cseg.copy(rseg)
        try:
            for i in range(cseg.length()):
                cseg.put1d(i,int(rmap[rseg.at1d(i)]))
        except IndexError:
            raise Exception("renumbering failed")

    # return everything we computed
    return Record(output=result,
                  raw=outs,
                  ins=as_intarray(ins),
                  outs=as_intarray(outs),
                  costs=costs,
                  rseg=rseg,
                  cseg=cseg,
                  lattice=lattice,
                  cost=costs.sum())


def compute_alignment(lattice,rseg,lmodel,beam=10000,verbose=0):
    """Given a lattice produced by a recognizer, a raw segmentation,
    and a language model, computes the best solution, the cseg, and
    the corresponding costs.  These are returned as Python data structures.
    The recognition lattice needs to have rseg's segment numbers as inputs
    (pairs of 16 bit numbers); SimpleGrouper produces such lattices."""

    # perform the beam search through the lattice and the model
    v1 = narray.intarray()
    v2 = narray.intarray()
    ins = narray.intarray()
    outs = narray.intarray()
    costs = narray.floatarray()
    ocropus.beam_search(v1,v2,ins,outs,costs,lattice,lmodel,beam)

    n = ins.length()
    assert n==outs.length()

    result = ""
    segs = []
    for i in range(n):
        if outs.at(i)<=0: continue
        start = ins.at(i)>>16
        end = ins.at(i)&0xffff
        result += chr(outs.at(i))
        segs.append((start,end))

    if len(segs)==0:
        print "???",n,segs
        cseg = None
    else:
        rmap = zeros(amax([s[1] for s in segs])+1,'i')
        for i in range(len(segs)):
            start,end = segs[i]
            if verbose: print i+1,start,end,"'%s'"%result[i],costs.at(i)
            if end==0: continue
            rmap[start:end+1] = i+1

        cseg = narray.intarray()
        cseg.copy(rseg)
        try:
            for i in range(cseg.length()):
                cseg.put1d(i,int(rmap[rseg.at1d(i)]))
        except IndexError:
            raise Exception("renumbering failed")

    return Record(output=result,
                  raw=outs,
                  ins=as_intarray(ins),
                  outs=as_intarray(outs),
                  segs=segs,
                  costs=costs,
                  rseg=rseg,
                  cseg=cseg,
                  lattice=lattice,
                  cost=costs.sum())

