### Components from the OCRopus native library that have been replaced
### by the new refactored libraries.

import os,os.path,re,numpy,unicodedata,sys,warnings,inspect,glob,traceback
import numpy
from numpy import *
from scipy.misc import imsave
from scipy.ndimage import interpolation,measurements,morphology
from common import *

class CleanupGray(CommonComponent):
    """Cleanup grayscale images."""
    c_interface = "ICleanupGray"
    def cleanup_gray(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.cleanup_gray(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class DeskewGrayPageByRAST(CleanupGray):
    """Page deskewing for gray scale images."""
    def __init__(self):
        self.make_("DeskewGrayPageByRAST")

class CleanupBinary(CommonComponent): 
    """Cleanup binary images."""
    c_interface = "ICleanupBinary"
    def cleanup(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.cleanup(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class RmHalftone(CleanupBinary):
    """Simple algorithm for removing halftones from binary images."""
    c_class = "RmHalftone"

class RmUnderline(CleanupBinary):
    """Simple algorithm for removing underlines from binary images."""
    c_class = "RmUnderLine"

class AutoInvert(CleanupBinary):
    """Simple algorithm for fixing inverted images."""
    c_class = "AutoInvert"

class DeskewPageByRAST(CleanupBinary):
    """Page deskewing for binary images."""
    c_class = "DeskewPageByRAST"

class RmBig(CleanupBinary):
    """Remove connected components that are too big to be text."""
    c_class = "RmBig"

class DocClean(CleanupBinary):
    """Remove document image noise components."""
    c_class = "DocClean"

class PageFrameByRAST(CleanupBinary):
    """Remove elements outside the document page frame."""
    c_class = "PageFrameByRAST"

class Binarize(CommonComponent):
    """Binarize images."""
    c_interface = "IBinarize"
    def binarize(self,page,type='f'):
        """Binarize the image; returns a tuple consisting of the binary image and
        a possibly transformed grayscale image."""
        if len(page.shape)==3: page = mean(page,axis=2)
        bin = iulib.bytearray()
        gray = iulib.bytearray()
        self.comp.binarize(bin,gray,page2narray(page,'B'))
        return (narray2page(bin,type=type),narray2page(gray,type=type))
    def binarize_color(self,page,type='f'):
        result = iulib.bytearray()
        self.comp.binarize_color(result,page2narray(page,'B'))
        return narray2page(result,type=type)

class StandardPreprocessing(Binarize):
    """Complete pipeline of deskewing, binarization, and page cleanup."""
    c_class = "StandardPreprocessing"

class BinarizeByRange(Binarize):
    """Simple binarization using the mean of the range."""
    c_class = "BinarizeByRange"

class BinarizeBySauvola(Binarize):
    """Fast variant of Sauvola binarization."""
    c_class = "BinarizeBySauvola"

class BinarizeByOtsu(Binarize):
    """Otsu binarization."""
    c_class = "BinarizeByOtsu"

class BinarizeByHT(Binarize):
    """Binarization by hysteresis thresholding."""
    c_class = "BinarizeByHT"

class TextImageClassification(CommonComponent):
    """Perform text/image classification."""
    c_interface = "ITextImageClassification"
    def textImageProbabilities(self,page):
        result = iulib.intarray()
        self.comp.textImageProbabilities(result,page2narray(page,'B'))
        return narray2pseg(result)

class SegmentPage(CommonComponent):
    """Segment a page into columns and lines (layout analysis)."""
    c_interface = "ISegmentPage"
    def segment(self,page,obstacles=None,black=0):
        page = page2narray(page,'B')
        # iulib.write_image_gray("_seg_in.png",page)
        result = iulib.intarray()
        if obstacles not in [None,[]]:
            raise Unimplemented()
        else:
            self.comp.segment(result,page)
        if black: ocropus.make_page_segmentation_black(result)
        # iulib.write_image_packed("_seg_out.png",result)
        return narray2pseg(result)

class SegmentPageByRAST(SegmentPage):
    """Segment a page into columns and lines using the RAST algorithm."""
    c_class = "SegmentPageByRAST"

class SegmentPageByRAST1(SegmentPage):
    """Segment a page into columns and lines using the RAST algorithm,
    assuming there is only a single column.  This is more robust for
    single column documents than RAST."""
    c_class = "SegmentPageByRAST1"

class SegmentPageBy1CP(SegmentPage):
    """A very simple page segmentation algorithm assuming a single column
    document and performing projection."""
    c_class = "SegmentPageBy1CP"

class SegmentPageByXYCUTS(SegmentPage):
    """An implementation of the XYCUT layout analysis algorithm.  Not
    recommended for production use."""
    c_class = "SegmentPageByXYCUTS"

class SegmentLine(CommonComponent):
    """Segment a line into character parts."""
    c_interface = "ISegmentLine"
    def charseg(self,line):
        """Segment a text line into potential character parts."""
        result = iulib.intarray()
        self.comp.charseg(result,line2narray(line,'B'))
        ocropus.make_line_segmentation_black(result)
        iulib.renumber_labels(result,1)
        return narray2lseg(result)

class DpLineSegmenter(SegmentLine):
    """Segment a text line by dynamic programming."""
    c_class = "DpSegmenter"

class SkelLineSegmenter(SegmentLine):
    """Segment a text line by thinning and segmenting the skeleton."""
    c_class = "SkelSegmenter"

class GCCSLineSegmenter(SegmentLine):
    """Segment a text line by connected components only, then grouping
    vertically related connected components."""
    c_class = "SegmentLineByGCCS"

class CCSLineSegmenter(SegmentLine):
    """Segment a text line by connected components only."""
    c_class = "ConnectedComponentSegmenter"

