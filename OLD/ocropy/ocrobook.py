# let the grouper handle ground truth and ground truth segmentation
# let the feature extractor handle size normalization etc

import sys,os,re,glob,math,glob
import iulib,ocropus
import traceback
from pylab import *
from utils import Record

default_segmenter = ocropus.make_ISegmentLine(os.getenv("segmenter") or "DpSegmenter")
default_grouper = ocropus.make_IGrouper("SimpleGrouper")
default_classifier = None

# Convert an iulib array to numpy
def N(image):
    """Convert an iulib array to a numpy image."""
    return iulib.numpy(image)

def NI(image):
    """Convert an iulib array to a numpy image (changes the coordinates)."""
    return transpose(N(image))[::-1,...]


def segmentation_correspondences(segmentation,cseg):
    """Compute the correspondences between the cseg and the rseg.
    This simply computes which rseg segments a cseg segment
    overlaps.
    FIXME make this a little more robust in case the cseg
    and rseg are slightly different around the edges"""
    a = ravel(array(N(segmentation),'i'))
    b = ravel(array(N(cseg),'i'))
    h = {}
    for i in range(len(a)):
        if b[i]==0: continue
        if a[i]==0: continue
        h[b[i]] = h.get(b[i],{})
        h[b[i]][a[i]] = 1
    n = amax(h.keys())+1
    result = [None]*n
    for i in h.keys():
        result[i] = tuple(sorted(h[i].keys()))
    return result

def load_gt(file):
    """check for the presence of cseg and txt file
    and use them to label characters if available"""
    cfile = re.sub(r'\.png$','.cseg.gt.png',file)
    tfile = re.sub(r'\.png$','.gt.txt',file)
    if not os.path.exists(cfile):
        cfile = re.sub(r'\.png$','.cseg.png',file)
        tfile = re.sub(r'\.png$','.txt',file)
    if os.path.exists(cfile):
        cseg = iulib.intarray()
        iulib.read_image_packed(cseg,cfile)
        ocropus.make_line_segmentation_black(cseg)
    else:
        cseg = None
    if os.path.exists(tfile):
        text = open(tfile).read()
    else:
        text = None
    return cseg,text

class BadTranscript(Exception):
    def __init__(self,*args):
        Exception.__init__(self,*args)

class NoException(Exception):
    pass

def cseg_chars(files,suffix="gt",segmenter=None,grouper=None,has_gt=1,verbose=0):
    """Iterate through the characters contained in a cseg file.
    Argument should be a list of image files.  Given "line.png",
    uses "line.cseg.gt.png" and "line.gt.txt" if suffix="gt".
    Returns an iterator of raw,mask,cls. Attempts to align
    with ground truth unless has_gt=0."""
    # also accept individual files
    if type(files)==type(""):
        files = [files]
    # if no grouper is given, just instantiate a simple grouper
    if not grouper:
        grouper = ocropus.make_IGrouper("SimpleGrouper")
        grouper.pset("maxrange",1)
    # allow empty suffix as a special case
    if suffix is None:
        suffix = ""
    if suffix!="":
        suffix = "."+suffix
    # now iterate through all the image files
    for file in files:
        if verbose:
            print "# loading",file
        try:
            # load the text line
            image = iulib.bytearray()
            iulib.read_image_gray(image,file)
            base = re.sub("\.png$","",file)
            # load segmentation ground truth
            cseg_file = base+".cseg"+suffix+".png"
            print file,cseg_file
            cseg = iulib.intarray()
            if not os.path.exists(cseg_file):
                raise IOError(cseg_file)
            iulib.read_image_packed(cseg,cseg_file)
            ocropus.make_line_segmentation_black(cseg)
            # load text
            if has_gt:
                text_file = base+suffix+".txt"
                text = open(text_file).read()
                if text[-1]=="\n": text = text[:-1]
                if len(text)>iulib.max(cseg):
                    text = re.sub(r'\s+','',text)
                utext = iulib.ustrg()
                utext.assign(text) # FIXME should do UTF8 or u""
                if utext.length()!=iulib.max(cseg):
                    raise BadTranscript("mismatch transcript %d maxseg %d"%(utext.length(),iulib.max(cseg)))
                if verbose:
                    print "#",utext.length(),iulib.max(cseg)
            # perform the segmentation
            segmentation = iulib.intarray()
            if segmenter:
                segmenter.charseg(segmentation,image)
                ocropus.make_line_segmentation_black(segmentation)
                iulib.renumber_labels(segmentation,1)
            else:
                segmentation.copy(cseg)

            # invert the image, since that's the way we're doing
            # all the remaining processing
            iulib.sub(255,image)

            # set the segmentation in preparation for loading
            if has_gt:
                grouper.setSegmentationAndGt(segmentation,cseg,utext)
            else:
                grouper.setSegmentation(segmentation)

            # now iterate through the segments of the line
            for i in range(grouper.length()):
                cls = None
                if has_gt:
                    cls = grouper.getGtClass(i)
                    if cls==-1:
                        cls = ""
                    else:
                        cls = chr(cls)
                raw = iulib.bytearray()
                mask = iulib.bytearray()
                grouper.extractWithMask(raw,mask,image,i,1)
                # print "component",i,N(segments),amax(N(raw)),raw.dim(0),raw.dim(1)
                # imshow(NI(raw)); gray(); show()
                yield Record(raw=raw,mask=mask,cls=cls,index=i,
                             bbox=grouper.boundingBox(i))
        except IOError,e:
            raise e
        except:
            print "FAILED",sys.exc_info()[0]
            print traceback.format_exc()
            continue

def chars_no_gt(files,segmenter=default_segmenter,grouper=default_grouper):
    for file in files:
        print "# loading",file
        image = iulib.bytearray()
        try:
            iulib.read_image_gray(image,file)
            segmentation = iulib.intarray()
            segmenter.charseg(segmentation,image)
            ocropus.make_line_segmentation_black(segmentation)
            iulib.renumber_labels(segmentation,1)
            grouper.setSegmentation(segmentation)
            iulib.sub(255,image)
            for i in range(grouper.length()):
                cls = None
                raw = iulib.bytearray()
                mask = iulib.bytearray()
                grouper.extractWithMask(raw,mask,image,i,1)
                yield raw,mask,cls
        except NoException:
            print "FAILED",sys.exc_info()[0]
            continue

