import re,sys,os
from pylab import *
import unittest
import ocrolib
import numpy
from ocrolib import fgen,mlp,fstutils
from ocrolib.fstutils import mkfst,printfst
from ocropus import L_SIGMA,L_RHO,L_PHI,L_EPSILON


def EQ(a,b):
    if type(a) in [numpy.ndarray,list] and type(b) in [numpy.ndarray,list]:
        a = array(a)
        b = array(b)
        if a.shape!=b.shape: return False
        return (a==b).all()
    return a==b

class TestFsts(unittest.TestCase):
    def testBestpath(self):
        fst = mkfst(3,[(0,1,"a"),(1,2,"b"),(2,3,"c")])
        assert fst.bestpath()=="abc"
        fst = mkfst(3,[(0,1,"a",1.0),(1,2,"b",1.0),(2,3,"c",1.0),(1,2,"x",0.3)])
        assert fst.bestpath()=="axc"

    # simple acceptor

    def testBeamsearch(self):
        fst1 = mkfst(3,[(0,1,1001,1.0),(1,2,1002,2.0),(2,3,1003,3.0)])
        fst2 = mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0),(2,3,1003,30.0)])
        (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
        assert EQ(costs,[0.0,11.0,22.0,33.0,0.0]),costs
        assert EQ(v1,[0,1,2,3]),v1
        assert EQ(v2,[0,1,2,3]),v2
        assert EQ(ins,[0,1001,1002,1003]),ins
        assert EQ(outs,[0,1001,1002,1003]),outs

    # cycle against linear

    def testBeamsearchCycle(self):
        fst1 = mkfst(0,[(0,0,1001,1.0),(0,0,1002,2.0),(0,0,1003,3.0)])
        fst2 = mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0),(2,3,1003,30.0)])
        (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
        assert EQ(costs,[0.0,11.0,22.0,33.0,0.0]),costs
        assert EQ(v1,[0,0,0,0]),v1
        assert EQ(v2,[0,1,2,3]),v2
        assert EQ(ins,[0,1001,1002,1003]),ins
        assert EQ(outs,[0,1001,1002,1003]),outs

  
    # try different input/output labels

    def testBeamsearchLabels(self):
        fst1 = mkfst(3,[(0,1,1001,1.0,91),(1,2,1002,2.0,92),(2,3,1003,3.0,93)])
        fst2 = mkfst(3,[(0,1,101,10.0,1001),(1,2,102,20.0,1002),(2,3,103,30.0,1003)])
        (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
        assert EQ(costs,[0.0,11.0,22.0,33.0,0.0]),costs
        assert EQ(v1,[0,1,2,3]),v1
        assert EQ(v2,[0,1,2,3]),v2
        assert EQ(ins,[0,91,92,93]),ins
        assert EQ(outs,[0,101,102,103]),outs

    # try an epsilon transition

    def testBeamsearchEps(self):
        fst1 = mkfst(4,[(0,1,1001,1.0),(1,2,1002,2.0),(2,3,L_EPSILON,3.0),(3,4,1003,4.0)])
        fst2 = mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0),(2,3,1003,30.0)])
        result = ocrolib.beam_search(fst1,fst2,1000)
        (v1,v2,ins,outs,costs) = result
        assert EQ(costs,[0.0,11.0,22.0,3.0,34.0,0.0]),costs
        assert EQ(v1,[0,1,2,3,4]),v1
        assert EQ(v2,[0,1,2,2,3]),v2
        assert EQ(ins,[0,1001,1002,0,1003]),ins
        assert EQ(outs,[0,1001,1002,0,1003]),outs

    # try rhos

    def testBeamsearchRho(self):
        fst1 = mkfst(3,[(0,1,1001,1.0),(1,2,1002,2.0),(2,3,1003,3.0)])
        # L_RHO should behave like the matching symbol, so we test both
        fst2s = [("with 1002",mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0,1002),(2,3,1003,30.0)])),
                 ("with rho",mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0,L_RHO),(2,3,1003,30.0)]))]
        for key,fst2 in fst2s:
            (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
            assert EQ(costs,[0.0,11.0,22.0,33.0,0.0]),costs
            assert EQ(v1,[0,1,2,3]),v1
            assert EQ(v2,[0,1,2,3]),v2
            assert EQ(ins,[0,1001,1002,1003]),ins
            assert EQ(outs,[0,1001,1002,1003]),outs

    # try phis (FAILS, FIXME)

    def testBeamsearchPhi(self):
        fst1 = mkfst(1,[(0,1,1001,1.0)])
        # L_PHI should behave like L_EPSILON on failure to match, so we test both
        fst2s = [("with epsilon",mkfst(3,[(0,1,9999,0.0),(0,2,L_EPSILON,0.0),(2,3,1001,0.0)])),
                 ("with phi",mkfst(3,[(0,1,9999,0.0),(0,2,L_PHI,0.0),(2,3,1001,0.0)]))]
        for key,fst2 in fst2s:
            (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
            assert EQ(costs,[0.0,0.0,1.0,0.0]),"%s: %s"%(key,costs)
            assert EQ(v1,[0,0,1]),v1
            assert EQ(v2,[0,2,3]),v2
            assert EQ(ins,[0,0,1001]),ins
            assert EQ(outs,[0,0,1001]),outs

    # try sigmas (FAILS, FIXME)

    def testBeamsearchSigma(self):
        fst1 = mkfst(3,[(0,1,1001,1.0),(1,2,1002,2.0),(2,3,1003,3.0)])
        # L_SIGMA should behave like the matching symbol, so we test both
        fst2s = [("with 1002",mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0,1002),(2,3,1003,30.0)])),
                 ("with sigma",mkfst(3,[(0,1,1001,10.0),(1,2,1002,20.0,L_SIGMA),(2,3,1003,30.0)]))]
        for key,fst2 in fst2s:
            (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
            assert EQ(costs,[0.0,11.0,22.0,33.0,0.0]),"%s: %s"%(key,costs)
            assert EQ(v1,[0,1,2,3]),v1
            assert EQ(v2,[0,1,2,3]),v2
            assert EQ(ins,[0,1001,1002,1003]),ins
            assert EQ(outs,[0,1001,1002,1003]),outs

    # try various special labels against each other
    
    def testBeamsearchRhoRho(self):
        fst1 = mkfst(3,[(0,1,1001,1.0),(1,2,L_RHO,2.0),(2,3,1003,3.0)])
        fst2 = mkfst(3,[(0,1,1001,10.0),(1,2,L_RHO,20.0),(2,3,1003,30.0)])
        result = ocrolib.beam_search(fst1,fst2,1000)
        (v1,v2,ins,outs,costs) = result
        assert EQ(costs,[0.0,11.0,22.0,33.0,0.0]),costs
        assert EQ(v1,[0,1,2,3]),v1
        assert EQ(v2,[0,1,2,3]),v2
        assert EQ(ins,[0,1001,L_RHO,1003]),ins
        assert EQ(outs,[0,1001,L_RHO,1003]),outs

    def testBeamsearchRhoEps(self):
        fst1 = mkfst(3,[(0,1,1001,1.0),(1,2,L_EPSILON,2.0),(2,3,1003,3.0)])
        fst2 = mkfst(3,[(0,1,1001,10.0),(1,2,L_RHO,20.0),(2,3,1003,30.0)])
        (v1,v2,ins,outs,costs) = ocrolib.beam_search(fst1,fst2,1000)
        assert EQ(costs,[0.0,Inf]),costs

        

if __name__=="__main__":
    unittest.main()
        
