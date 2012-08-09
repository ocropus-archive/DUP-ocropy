################################################################
### functions specific to text line processing
### (text line segmentation is in lineseg)
################################################################

import sys,os,re,glob,math,glob,signal
import scipy
from scipy import stats
from scipy.ndimage import measurements,interpolation,morphology,filters
from pylab import *
import common,sl,morph
from toplevel import *

################################################################
### line segmentation geometry estimates based on
### segmentations
################################################################

seg_geometry_display = 0
geowin = None
geoax = None

@checks(SEGMENTATION,math=BOOL)
def seg_geometry(segmentation,math=1):
    """Given a line segmentation (either an rseg--preferably connected
    component based--or a cseg, return (mh,a,b), where mh is the
    medium component height, and y=a*x+b is a line equation (in
    Postscript coordinates) for the center of the text line.  This
    function is used as a simple, standard estimator of text line
    geometry.  The intended use is to encode the size and centers of
    bounding boxes relative to these estimates and add these as
    features to the input of a character classifier, allowing it to
    distinguish otherwise ambiguous pairs like ,/' and o/O."""
    boxes = seg_boxes(segmentation,math=math)
    heights = [(y1-y0) for (y0,y1,x0,x1) in boxes]
    mh = stats.scoreatpercentile(heights,per=40)
    centers = [(avg(y0,y1),avg(x0,x1)) for (y0,y1,x0,x1) in boxes]
    xs = array([x for y,x in centers])
    ys = array([y for y,x in centers])
    a,b = polyfit(xs,ys,1)
    if seg_geometry_display:
        print "seggeo",math
        from matplotlib import patches
        global geowin,geoax
        old = gca()
        if geowin is None:
            geowin = figure()
            geoax = geowin.add_subplot(111)
        geoax.cla()
        geoax.imshow(segmentation!=0,cmap=cm.gray)
        for (y0,y1,x0,x1) in boxes:
            p = patches.Rectangle((x0,y0),x1-x0,y1-y0,edgecolor="red",fill=0)
            geoax.add_patch(p)
        xm = max(xs)
        geoax.plot([0,xm],[b,a*xm+b],'b')
        geoax.plot([0,xm],[b-mh/2,a*xm+b-mh/2],'y')
        geoax.plot([0,xm],[b+mh/2,a*xm+b+mh/2],'y')
        geoax.plot(xs,[y for y in ys],"g.")
        sca(old)
        print "mh",mh,"a",a,"b",b
    return mh,a,b

def avg(*args):
    return mean(args)

@deprecated
def rel_char_geom(box,params):
    """Given a character bounding box and a set of line geometry parameters,
    compute relative character position and size."""
    y0,y1,x0,x1 = box
    assert y1>y0 and x1>x0,"%s %s"%((x0,x1),(y0,y1))
    mh,a,b = params
    y = avg(y0,y1)
    x = avg(x0,x1)
    yl = a*x+b
    rel_ypos = (y-yl)/mh
    rel_width = (x1-x0)*1.0/mh
    rel_height = (y1-y0)*1.0/mh
    # ensure some reasonable bounds
    assert rel_ypos>-100 and rel_ypos<100
    assert rel_width>0 and rel_width<100
    assert rel_height>0 and rel_height<100
    return rel_ypos,rel_width,rel_height

@deprecated
def rel_geo_normalize(rel):
    """Given a set of geometric parameters, normalize them into the
    range -1...1 so that they can be used as input to a neural network."""
    if rel is None: return None
    if type(rel)==str:
        rel = [float(x) for x in rel.split()]
    ry,rw,rh = rel
    if not (rw>0 and rh>0): return None
    ry = clip(2*ry,-1.0,1.0)
    rw = clip(log(rw),-1.0,1.0)
    rh = clip(log(rh),-1.0,1.0)
    geometry = array([ry,rw,rh],'f')
    return geometry

@deprecated
def seg_boxes(seg,math=0):
    """Given a color segmentation, return a list of bounding boxes.
    Bounding boxes are returned as tuples (y0,y1,x0,x1).  With
    math=0, raster coordinates are used, with math=1, Postscript
    coordinates are used (however, the order of the values in the
    tuple doesn't change)."""
    seg = array(seg,'uint32')
    slices = morph.find_objects(seg)
    h = seg.shape[0]
    result = []
    for i in range(len(slices)):
        if slices[i] is None: continue
        (ys,xs) = slices[i]
        if math:
            result += [(h-ys.stop-1,h-ys.start-1,xs.start,xs.stop)]
        else:
            result += [(ys.start,ys.stop,xs.start,xs.stop)]
    return result

################################################################
### image based estimation of line geometry, as well
### as dewarping
################################################################

@checks(DARKLINE)
def estimate_baseline(line):
    """Compute the baseline by fitting a polynomial to the gradient.
    TODO: use robust fitting, special case very short line, limit parameter ranges"""
    line = line*1.0/amax(line)
    vgrad = morphology.grey_closing(line,(1,40))
    vgrad = filters.gaussian_filter(vgrad,(2,60),(1,0))
    if amin(vgrad)>0 or amax(vgrad)<0: raise BadLine()
    h,w = vgrad.shape
    baseline = fitext(vgrad)
    return baseline

@checks(DARKLINE)
def dewarp_line(line,show=0):
    """Dewarp the baseline of a line based in estimate_baseline.
    Returns the dewarped image."""
    line = line*1.0/amax(line)
    line = r_[zeros(line.shape),line]
    h,w = line.shape
    baseline = estimate_baseline(line)
    ys = polyval(baseline,arange(w))
    base = 2*h/3
    temp = zeros(line.shape)
    for x in range(w):
        temp[:,x] = interpolation.shift(line[:,x],(base-ys[x]),order=1)
    return temp

    line = line*1.0/amax(line)

@checks(DARKLINE)
def estimate_xheight(line,scale=1.0,debug=0):
    """Estimates the xheight of a line based on image processing and
    filtering."""
    vgrad = morphology.grey_closing(line,(1,int(scale*40)))
    vgrad = filters.gaussian_filter(vgrad,(2,int(scale*60)),(1,0))
    if amin(vgrad)>0 or amax(vgrad)<0: raise Exception("bad line")
    if debug: imshow(vgrad)
    proj = sum(vgrad,1)
    proj = filters.gaussian_filter(proj,0.5)
    top = argmax(proj)
    bottom = argmin(proj)
    return bottom-top,bottom

@checks(DARKLINE)
def latin_mask(line,scale=1.0,r=1.2,debug=0):
    """Estimate a mask that covers letters and diacritics of a text
    line for Latin alphabets."""
    vgrad = morphology.grey_closing(1.0*line,(1,int(scale*40)))
    vgrad = filters.gaussian_filter(vgrad,(2,int(scale*60)),(1,0))
    tops = argmax(vgrad,0)
    bottoms = argmin(vgrad,0)
    mask = zeros(line.shape)
    xheight = mean(bottoms-tops)
    for i in range(len(bottoms)):
        d = bottoms[i]-tops[i]
        y0 = int(maximum(0,bottoms[i]-r*d))
        mask[y0:bottoms[i],i] = 1
    return mask

@checks(DARKLINE)
def latin_filter(line,scale=1.0,r=1.5,debug=0):
    """Filter out noise from a text line in Latin alphabets."""
    bin = (line>0.5*amax(line))
    mask = latin_mask(bin,scale=scale,r=r,debug=debug)
    mask = morph.keep_marked(bin,mask)
    mask = filters.maximum_filter(mask,3)
    return line*mask

@checks(DARKLINE)
def remove_noise(line,minsize=8):
    """Remove small pixels from an image."""
    bin = (line>0.5*amax(line))
    labels,n = morph.label(bin)
    sums = measurements.sum(bin,labels,range(n+1))
    sums = sums[labels]
    good = minimum(bin,1-(sums>0)*(sums<minsize))
    return good
    
