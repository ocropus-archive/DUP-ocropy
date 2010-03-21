#!/usr/bin/python

import iulib,ocropus
from pylab import *
from scipy.stats.stats import trim1
from scipy.ndimage import measurements
from scipy.misc import imsave

def quick_check_page_components(image,dpi=200,min_pt=9,max_pt=18):
    """Check whether the given page image roughly conforms to the kinds of
    inputs that OCRopus accepts."""

    w = image.dim(0)
    h = image.dim(1)

    ## currently don't deal with very low or very high resolution images
    assert dpi>=200 and dpi<=600,\
        "[error] resolution should be between 200 dpi and 600 dpi"

    ## this is called for checking page images; page images should have
    ## a minimum and maximum size (no blocks, text lines, or newspaper spreads)
    if w<3*dpi or w>10*dpi:
        print "[warning] image width %g in (%d px, %d dpi)"%(w/dpi,w,dpi)
    elif h<3*dpi or h>10*dpi:
        print "[warning] image height %g in (%d px, %d dpi)"%(h/dpi,h,dpi)

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
    min_px = (1.0/3.0) * min_pt*dpi/72.0
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

    ## print some information
    print "# min",min_px,"max",max_px
    density = n*dpi**2*1.0/w/h
    print "# normal",sum(normal),"small",sum(small),"large",sum(large)
    print "# density",density
    print "# covered",covered

    ## compute aspect ratio statistics; we're using a right-trimmed mean of
    ## biggish components; this means that we exclude characters like "-"
    ## from the computation (because they are not biggish), and we also exclude
    ## large connected components such as rules (since they are trimmed off)
    ## the remaining mean should represent the mean of connected components that
    ## make up the bulk of the text on the page
    aspect = heights*1.0/widths
    aspect = aspect[biggish]
    a_mean = mean(trim1(aspect,0.1,tail='right'))

    ## print warning messages for unusually low or high aspect ratios
    if len(aspect)>100:
        print "# aspect mean",a_mean
        if a_mean<1.0:
            print "[note] unusually low mean aspect ratio"
            print "[note] this may be due to touching characters (threshold too low)"
            print "[note] and cause OCR to give poor results"
        elif a_mean>1.4:
            print "[note] unusually high mean aspect ratio"
            print "[note] this may be due to broken characters (threshold too high)"
            print "[note] and cause OCR to give poor results"

    ## print warning messages related to noise and/or resolution
    if covered<0.05 or density<8:
        print "[note] page doesn't contain a lot of text"
    elif sum(normal)<5*sum(small):
        print "[warning] too many small components"
        print "[warning] possible causes:"
        print "[warning] - maybe text is too low resolution"
        print "[warning] - maybe image was thresholded too high"
        print "[warning] - maybe text image segmentation failed"
        print "[warning] - maybe the image contains a lot of '.' or diacritics"
        print "[warning] OCR may not work well on this page depending on the cause"
        print "[warning] OCRopus does not recognize screen shots or camera captured text"
    elif sum(large)>sum(normal):
        print "[warning] too many large components"
        print "[warning] your image may have too high resolution or too large a font size"
        print "[warning] OCR will likely not work well"

