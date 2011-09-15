################################################################
### Miscellaneous plotting functions based on matplotlib.
################################################################

import matplotlib 
matplotlib.use('GTK') 
import __init__ as ocropy # FIXME
from matplotlib import patches
from pylab import *
import re
import common

def gtk_yield():
    import gtk
    while gtk.events_pending():
       gtk.main_iteration(False)

def draw_pseg(pseg,axis=None):
    if axis is None:
        axis = subplot(111)
    h = pseg.dim(1)
    regions = ocropy.RegionExtractor()
    regions.setPageLines(pseg)
    for i in range(1,regions.length()):
        x0,y0,x1,y1 = (regions.x0(i),regions.y0(i),regions.x1(i),regions.y1(i))
        p = patches.Rectangle((x0,h-y1-1),x1-x0,y1-y0,edgecolor="red",fill=0)
        axis.add_patch(p)
    
def draw_linerec(image,fst,rseg,lmodel,axis=None):
    raise Error("FIXME draw_linerec")
    if type(image)==numpy.ndarray: image = common.numpy2narray(image)
    if axis is None:
        axis = subplot(111)
    axis.imshow(image,cmap=cm.gray)
    result,cseg,costs = common.compute_alignment(fst,rseg,lmodel)
    ocropy.make_line_segmentation_black(cseg)
    ocropy.renumber_labels(cseg,1)
    bboxes = ocropy.rectarray()
    ocropy.bounding_boxes(bboxes,cseg)
    s = re.sub(r'\s+','',result)
    h = image.dim(1)
    for i in range(1,bboxes.length()):
        r = bboxes.at(i)
        x0,y0,x1,y1 = (r.x0,r.y0,r.x1,r.y1)
        p = patches.Rectangle((x0,h-y1-1),x1-x0,y1-y0,edgecolor=(0.0,0.0,1.0,0.5),fill=0)
        axis.add_patch(p)
        if i>0 and i-1<len(s):
            axis.text(x0,h-y0-1,s[i-1],color="red",weight="bold",fontsize=14)
    draw()

def draw_aligned(result,axis=None):
    raise Error("FIXME draw_aligned")
    if axis is None:
        axis = subplot(111)
    axis.imshow(NI(result.image),cmap=cm.gray)
    cseg = result.cseg
    if type(cseg)==numpy.ndarray: cseg = common.lseg2narray(cseg)
    ocropy.make_line_segmentation_black(cseg)
    ocropy.renumber_labels(cseg,1)
    bboxes = ocropy.rectarray()
    ocropy.bounding_boxes(bboxes,cseg)
    s = re.sub(r'\s+','',result.output)
    h = cseg.dim(1)
    for i in range(1,bboxes.length()):
        r = bboxes.at(i)
        x0,y0,x1,y1 = (r.x0,r.y0,r.x1,r.y1)
        p = patches.Rectangle((x0,h-y1-1),x1-x0,y1-y0,edgecolor=(0.0,0.0,1.0,0.5),fill=0)
        axis.add_patch(p)
        if i>0 and i-1<len(s):
            axis.text(x0,h-y0-1,s[i-1],color="red",weight="bold",fontsize=14)
    draw()

def plotgrid(data,d=10,shape=(30,30)):
    """Plot a list of images on a grid."""
    ion()
    gray()
    clf()
    for i in range(min(d*d,len(data))):
        subplot(d,d,i+1)
        row = data[i]
        if shape is not None: row = row.reshape(shape)
        imshow(row)
    ginput(1,timeout=0.1)

