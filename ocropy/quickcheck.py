#!/usr/bin/python

import iulib,ocropus
from pylab import *
from scipy.stats.stats import trim1
from scipy.ndimage import measurements
from scipy.misc import imsave
from utils import NI,N

def alert(*args):
    sys.stderr.write(" ".join([str(x) for x in args]))
    sys.stderr.write("\n")

class Record:
    def __init__(self,**kw):
        for k in kw.keys():
            self.__dict__[k] = kw[k]

def cc_statistics(image,dpi,min_pt,max_pt,verbose=0):
    w = image.dim(0)
    h = image.dim(1)

    ## compute connected component widths and heights
    components = iulib.intarray()
    components.copy(image)
    iulib.sub(iulib.max(components),components)
    iulib.label_components(components)
    boxes = iulib.rectarray()
    iulib.bounding_boxes(boxes,components)
    n = boxes.length()
    widths = array([boxes.at(i).width() for i in range(1,n)])
    heights = array([boxes.at(i).height() for i in range(1,n)])

    ## we consider "normal" components that are between 1/3 of the 
    ## size of the minimum sized font and the full size of the 
    ## maxmimum sized font; to compute this, we need to convert from
    ## font sizes in pt to pixel sizes, using the given dpi
    maxs = maximum(widths,heights)
    min_px_em = min_pt*dpi/72.0
    min_px = (1.0/3.0) * min_px_em
    max_px = max_pt*dpi/72.0

    ## compute the total page area covered by bounding boxes of connected
    ## components (we don't bother to try to avoid double counts in overlapping boxes)
    covered = sum(widths*heights)*1.0/w/h

    ## small components are those whose maximum dimension is smaller that the min size
    small = (maxs<min_px)

    ## large components have at least one dimension better than the max size
    large = (maxs>max_px)

    ## biggish components have both dimensions bigger than the small size (this
    ## excludes "." and "-" and is used for aspect ratio computations)
    biggish = ((widths>min_px)&(heights>min_px))

    ## normal boxes are those that are neither small nor large
    normal = ~(small|large)

    ## absolute density of characters per square inch
    density = n*dpi**2*1.0/w/h

    ## relative density of characters per em
    h_density = n/(w/min_px_em)

    ## print some information
    if verbose:
        alert("# min",min_px,"max",max_px)
        alert("# normal",sum(normal),"small",sum(small),"large",sum(large))
        alert("# density",density)
        alert("# h_density",h_density)
        alert("# covered",covered)


    ## compute aspect ratio statistics; we're using a right-trimmed mean of
    ## biggish components; this means that we exclude characters like "-"
    ## from the computation (because they are not biggish), and we also exclude
    ## large connected components such as rules (since they are trimmed off)
    ## the remaining mean should represent the mean of connected components that
    ## make up the bulk of the text on the page
    aspect = heights*1.0/widths
    aspect = aspect[biggish]
    a_mean = mean(trim1(aspect,0.1,tail='right'))

    result = Record(
        biggish = sum(biggish),
        normal = sum(normal),
        small = sum(small),
        large = sum(large),
        density = density,
        h_density = h_density,
        a_mean = a_mean,
        covered=covered,
    )

    return result


def quick_check_page_components(image,dpi=200,min_pt=9,max_pt=18,verbose=0):
    """Check whether the given page image roughly conforms to the kinds of
    inputs that OCRopus accepts. Returns a value between 0 and 1, with 1 meaning
    OK and 0 meaning definitely reject"""

    status = 1.0

    ## currently don't deal with very low or very high resolution images
    assert dpi>=200 and dpi<=600,\
        "[error] resolution should be between 200 dpi and 600 dpi"

    ## this is called for checking page images; page images should have
    ## a minimum and maximum size (no blocks, text lines, or newspaper spreads)
    w = image.dim(0)
    h = image.dim(1)
    if w<3*dpi or w>10*dpi:
        alert("[warn] image width %g in (%d px, %d dpi)"%(w/dpi,w,dpi))
    elif h<3*dpi or h>10*dpi:
        alert("[warn] image height %g in (%d px, %d dpi)"%(h/dpi,h,dpi))

    p = cc_statistics(image,dpi,min_pt,max_pt,verbose=verbose)

    ## alert(warning messages for unusually low or high aspect ratios)
    if p.biggish>100:
        if verbose:
            alert("# aspect mean",p.a_mean)
        if p.a_mean<1.0:
            alert("[note] unusually low mean aspect ratio (bad threshold?)")
            status = min(status,0.7)
        elif p.a_mean>1.4:
            alert("[note] unusually high mean aspect ratio (bad threshold?)")
            status = min(status,0.7)

    ## alert(warning messages related to noise and/or resolution)
    if p.covered<0.05 or p.density<8:
        alert("[note] page doesn't contain a lot of text")
        status = min(status,0.9)
    elif p.normal<5*p.small:
        alert("[warn] page has too many small components (%d/%d/%d) (wrong resolution? bad threshold?)" % (p.small,p.normal,p.large))
        status = min(status,0.6)
    elif p.large>p.normal:
        alert("[warn] page has too many large components (%d/%d/%d)" % (p.small,p.normal,p.large))
        status = min(status,0.6)
    return status

def quick_check_line_components(image,dpi=200,min_pt=9,max_pt=18,verbose=0):
    """Check whether the given line image roughly conforms to the kinds of
    inputs that OCRopus accepts."""

    status = 1.0

    ## currently don't deal with very low or very high resolution images
    assert dpi>=200 and dpi<=600,\
        "[error] resolution should be between 200 dpi and 600 dpi"

    p = cc_statistics(image,dpi,min_pt,max_pt,verbose=verbose)

    if p.normal==0:
        alert("[warn] no normal sized components found (%d/%d/%d)" % (p.small,p.normal,p.large))
        status = min(status,0.1)
    if p.large>p.normal:
        alert("[warn] components too large on average (%d/%d/%d)" % (p.small,p.normal,p.large))
        status = min(status,0.1)
    if p.density>100:
        alert("[warn] component density much too high (maybe halftone region?)")
        status = min(status,0.1)
    if p.h_density>2.0:
        alert("[warn] horizontal component density much too high (maybe halftone region?)")
        status = min(status,0.1)
    if p.normal<2*p.small and p.normal>5:
        alert("[warn] too many small components")
        status = min(status,0.6)
    if p.covered<0.3:
        alert("[warn] line contains a lot of empty space")
        status = min(status,0.9)
    return status

