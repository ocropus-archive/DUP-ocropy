#!/usr/bin/python
# -*- Encoding: utf-8 -*-

import re,sys,os
from pylab import *
import unittest
import ocrolib
from ocrolib import fgen,mlp,fstutils

class RecognitionWithExistingModel(unittest.TestCase):
    model_file = "2m2-reject.cmodel"
    LineRecognizer = ocrolib.CmodelLineRecognizer

    def __init__(self,*args):
        unittest.TestCase.__init__(self,*args)
        self.model = ocrolib.load_component(self.model_file)
        self.linerec = self.LineRecognizer(cmodel=self.model)

    def sequal(self,a,b):
        return re.sub(r'[^a-zA-Z]','',a)==re.sub(r'[^a-zA-Z]','',b)

    def testChars(self):
        for c in ["A","N","n"]:
            image = fgen.cairo_render_gray(c,fontname="Georgia",size=32)
            r = ocrolib.ocosts(self.model.coutputs(image))
            assert r[0][0]==c,r[:3]

    def testStrings(self):
        for s in ["woolly","mammoth"]:
            image = fgen.cairo_render_gray(s,fontname="Georgia",size=32)
            image = 1.0*(image < amax(image)/2)
            fst = self.linerec.recognizeLine(image)
            result = fst.bestpath()
            assert self.sequal(result,s),"result = %s"%result

    def testAlignment(self):
        s = "woolly"
        image = fgen.cairo_render_gray(s,fontname="Georgia",size=32)
        image = 1.0*(image < amax(image)/2)
        lattice,rseg = self.linerec.recognizeLineSeg(image)
        lmodel = fstutils.make_line_fst([s])
        result = ocrolib.compute_alignment(lattice,rseg,lmodel)
        assert result.output==s
        assert "".join(result.output_l)==s

    def testAlignmentWithLigatures(self):
        s = "woolly"
        gt = u"w_oo_lly"
        image = fgen.cairo_render_gray(s,fontname="Georgia",size=32)
        image = 1.0*(image < amax(image)/2)
        lattice,rseg = self.linerec.recognizeLineSeg(image)
        lmodel = fstutils.make_line_fst([gt])
        result = ocrolib.compute_alignment(lattice,rseg,lmodel)
        assert "".join(result.output_l)==s
        assert result.output_t==gt
        assert result.output_l==[u'w', u'oo', u'l', u'l', u'y']

class CharacterRecognizer(unittest.TestCase):
    Model = mlp.MlpModel

    def charimage(self,c):
        image = fgen.cairo_render_gray(c,fontname="Georgia",size=32)
        image = fgen.gauss_degrade(image)
        return image

    def testClassLabelExceptions(self):
        model = self.Model()
        # no spaces or control characters
        bad_classes = [" ","\r","\n"]
        # no high bit in plain ASCII string
        bad_classes += ["\x80","\x9f","\xff"]
        # no control characters or spaces in unicode strings
        bad_classes += [u" ",u"\07"]
        for bad in bad_classes:
            try:
                model.cadd(self.charimage("x"),bad)
            except ocrolib.BadClassLabel:
                pass
            else:
                self.fail("should throw bad class label exception")
    def testModelTraining(self):
        return
        for target in [u"x",u"ß",u"oo",u"中"]:
            model = self.Model(etas=[(0.1,5000)]*1,verbose=1)
            # train only one of each background class, this makes the
            # training fast and easy
            other = [u"a",u"b",u"c",u"d",u"e",u"zz"]
            for c in other:
                model.cadd(self.charimage(c),c)
            # train lots of the foreground task
            for i in range(100):
                model.cadd(self.charimage(target),target)
            model.updateModel()
            result = ocrolib.ocosts(model.coutputs(self.charimage(target)))
            assert result[0][0]==target

if __name__=="__main__":
    unittest.main()
