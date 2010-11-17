import sys,os,re,glob,math,glob,signal,cPickle
import iulib,ocropus
import components
from utils import N,NI,F,FI,Record,show_segmentation
from scipy.ndimage import interpolation,morphology,measurements
import scipy
import scipy.stats
from pylab import *
import unicodedata
import pickle
import __init__ as ocropy
import fstutils
import utils

def bestpath(lattice):
    s = ocropy.ustrg()
    lattice.bestpath(s)
    cost = 0.0
    return ocropy.ustrg_as_string(s)

def rect(r):
    return (r.x0,r.y0,r.x1,r.y1)

def floatimage(image):
    fimage = ocropy.floatarray()
    fimage.copy(image)
    ocropy.div(fimage,255.0)
    return fimage

class RejectException(Exception):
    def __init__(self,*args,**kw):
        super.__init__(*args,**kw)

# size classes of characters; note that "ij" are not here

class_x = "acemnorsuvwxz"
class_A = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt"
class_g = "gpqy"

# unambiguous size classes (e.g., "p" and "P" are too similar for
# unambiguous size determination)

class_xu = "aemnru"
class_Au = "023456789ABDEFGHKLMNQRTYbdfhkt"
class_gu = "gqy"

# confusions

# character pairs only distinguished by location
loc_confusions = [
    ["'",","],
    ["p","P"],
    ]

# character pairs only distinguished by size
size_confusions = [
    ["c","C"],
    ["o","O"],
    ["s","S"],
    ["u","U"],
    ["v","V"],
    ["w","W"],
    ["x","X"],
    ["z","Z"],
    ]

# characters that are smaller than x-height
small_height = [".",",","'",'"',"-"]

def guess_xheight_from_candidates(self,candidates):
    # for Latin script, use some information about character
    # classes
    heights = []
    for c in candidates:
        cls = c.outputs[0].cls
        if cls=="~": continue
        if cls in class_x: heights.append(c.bbox.height())
        elif cls in class_A: heights.append(c.bbox.height()/1.5)
        elif cls in class_g: heights.append(c.bbox.height()/1.5)
    if len(heights)>2:
        return median(heights)
    # fallback
    return median([c.bbox.height() for c in candidates if c.cls!="~"])

def estimate_xheight(candidates):
    short = []
    tall = []
    for x in candidates:
        x0,y0,x1,y1 = x.bbox.tuple()
        aspect = (y1-y0)*1.0/(x1-x0)
        if aspect<0.7:
            continue
        elif aspect<1.1:
            short.append(y1-y0)
        else:
            tall.append(y1-y0)
    if len(short)>4:
        scale = median(short)
    elif len(tall)>4:
        scale = median(tall)/1.4
    else:
        scale = None
    return scale

def ncost(x,params):
    m,v = params
    v *= 1.5
    c = -log(scipy.stats.norm(loc=m,scale=v).pdf(x))
    c = max(0.0,min(c,2.0))
    return c
def bestcost(x):
    return min([y[1] for y in x.outputs])

class SimpleLineModel:
    def __init__(self):
        pass
    def load(self,file):
        self.aspects = {}
        self.widths = {}
        self.heights = {}
        self.ys = {}
        table = {"a":self.aspects,"w":self.widths,"h":self.heights,"y":self.ys}
        with open(file,"r") as stream:
            for line in stream.readlines():
                line = line[:-1]
                which,char,count,mean,var = line.split()
                table[which][char] = (float(mean),float(var))
    def linecosts(self,candidates,image,cseg_guess=None,transcript_guess=None):
        """Given a list of character recognition candidates and their
        classifications, and an image of the corresponding text line,
        adjust the costs of the candidates to account for their position
        on the textline."""
        threshold = scipy.stats.scoreatpercentile([bestcost(x) for x in candidates],per=25)
        print "threshold",threshold
        best = [x for x in candidates if bestcost(x)<threshold]
        if len(best)<4: best = candidates
        base = median([x.bbox.y0 for x in best])
        scale = estimate_xheight(best)
        print "base",base,"scale",scale
        for x in candidates:
            x0,y0,x1,y1 = x.bbox.tuple()
            aspect = (y1-y0)*1.0/(x1-x0)
            costs = {}
            for cls,cost in x.outputs:
                costs[cls] = cost
            for a in costs:
                if a=="~": continue
                ac = ncost(aspect,self.aspects[a])
                if costs[a]<2 and ac>1.0:
                    print "adjusting",a,costs[a],ac,"aspect"
                costs[cls] += ac
            for a,b in loc_confusions:
                if abs(costs[a]-costs[b])>1.0: continue
                ac = ncost((y0-base)/scale,self.ys[a])
                bc = ncost((y0-base)/scale,self.ys[b])
                if costs[a]<10:
                    print "adjusting",a,b,costs[a],costs[b],"loc"
                if ac<bc: costs[b] += 2.0
                else: costs[a] += 2.0
            for a,b in size_confusions:
                if abs(costs[a]-costs[b])>1.0: continue
                ac = ncost((y1-y0)/scale,self.heights[a])
                bc = ncost((y1-y0)/scale,self.heights[b])
                if costs[a]<10:
                    print "adjusting",a,b,costs[a],costs[b],"size"
                if ac<bc: costs[b] += 2.0
                else: costs[a] += 2.0
            for a in small_height:
                ac = ncost((y1-y0)/scale,self.heights[a])
                if ac<1.0: continue
                if costs[a]<10:
                    print "penalizing",a,costs[a],ac,"small"
                costs[a] += ac
            x.outputs = [(cls,cost) for cls,cost in costs.items()]

class SimpleSpaceModel:
    def __init__(self):
        self.nonspace_cost = 4.0
        self.space_cost = 9999.0
        self.aspect_threshold = 2.0
        self.maxrange = 30
    def spacecosts(self,candidates,image):
        """Given a list of character recognition candidates and their
        classifications, and an image of the corresponding text line,
        compute a list of pairs of costs for putting/not putting a space
        after each of the candidate characters.

        The basic idea behind this simple algorithm is to try larger
        and larger horizontal closing operations until most of the components
        start having a "wide" aspect ratio; that's when characters have merged
        into words.  The remaining whitespace should be spaces.

        This is just a simple stopgap measure; it will be replaced with
        trainable space modeling."""

        w = image.dim(0)
        h = image.dim(1)
        image = ocropy.NI(image)
        image = 1*(image>0.5*(amin(image)+amax(image)))
        for r in range(0,self.maxrange):
            if r>0: closed = morphology.binary_closing(image,iterations=r)
            else: closed = image
            labeled,n = measurements.label(closed)
            objects = measurements.find_objects(labeled)
            aspects = []
            for i in range(len(objects)):
                s = objects[i]
                aspect = (s[1].stop-s[1].start)*1.0/(s[0].stop-s[0].start)
                aspects.append(aspect)
            maspect = median(aspects)
            if maspect>=self.aspect_threshold:
                break

        # close with a little bit of extra space
        closed = morphology.binary_closing(image,iterations=r+1)

        # compute the remaining aps
        gaps = (0==sum(closed,axis=0))
        morphology.binary_dilation(gaps,iterations=3)

        # every character box that ends near a cap gets a space appended
        result = []
        for c in candidates:
            if gaps[c.bbox.x1]:
                result.append((0.0,self.nonspace_cost))
            else:
                result.append((self.space_cost,0.0))

        return result

class LineRecognizer:
    # can't derive from IRecognizeLine -- breaks save/load (ocropus.IRecognizeLine)
    def __init__(self,cmodel=None,segmenter="DpSegmenter",best=None,
                 maxcost=None,reject_cost=None,minheight_letters=None):
        """Initialize a line recognizer that works from character models.
        The character shape model is given at initialization and needs to conform to
        the IModel interface.  The segmenter needs to support ISegmentLine.
        The best parameter determines how many of the top outputs from the classifier
        are used in the construction of the lattice.  The maxcost parameter is
        the maximum cost that will be assigned to a transiation in the lattice.
        The reject_cost is a cost above which a character won't get added to the lattice
        at all.  The minheight_letter threshold is the minimum height of a
        component (expressed as fraction of the medium segment height) in
        order to be added as a letter to the lattice."""

        self.set_defaults()
        self.cmodel = cmodel
        if best is not None: self.best = best
        if maxcost is not None: self.maxcost = maxcost
        if reject_cost is not None: self.reject_cost = reject_cost
        if minheight_letters is not None: self.min_height = minheight_letters

    def set_defaults(self):
        self.debug = 0
        self.segmenter = components.make_ISegmentLine("DpSegmenter")
        self.good_segmenter = components.make_ISegmentLine("SegmentLineByGCCS")
        # self.segmenter.pset("debug","dpsegmenter.png")
        # self.segmenter.pset("fix_diacritics",0)
        self.grouper = components.make_IGrouper("SimpleGrouper")
        self.cmodel = None
        self.best = 10
        self.maxcost = 30.0
        self.reject_cost = 10.0
        self.min_height = 0.5
        self.rho_scale = 1.0
        self.maxoverlap = 0.8
        self.spacemodel = SimpleSpaceModel()
        self.linemodel = None

    def info(self):
        for k in sorted(self.__dict__.keys()):
            print k,self.__dict__[k]
        self.cmodel.info()

    def recognizeLine(self,lattice,image):
        "Recognize a line, outputting a recognition lattice."""
        rseg = iulib.intarray()
        return self.recognizeLineSeg(lattice,rseg,image)

    def recognizeLineSeg(self,lattice,rseg,image):
        """Recognize a line.
        lattice: result of recognition
        rseg: intarray where the raw segmentation will be put
        image: line image to be recognized"""

        if self.debug: print "starting"

        ## increase segmentation scale for large lines
        h = image.dim(1)
        s = max(2.0,h/15.0)
        try:
            self.segmenter.pset("cost_smooth",s)
            if s>2.0: print "segmentation scale",s
        except: pass

        ## compute the raw segmentation
        if self.debug: print "segmenting"
        self.segmenter.charseg(rseg,image)
        if self.debug: print "done"
        ocropus.make_line_segmentation_black(rseg)
        if self.debug:
            print "here"
            clf()
            subplot(4,1,1)
            show_segmentation(rseg)
            draw()
            print "there"
        iulib.renumber_labels(rseg,1)
        self.grouper.setSegmentation(rseg)

        # compute the median segment height
        heights = []
        for i in range(self.grouper.length()):
            bbox = self.grouper.boundingBox(i)
            heights.append(bbox.height())
        mheight = median(array(heights))
        self.mheight = mheight

        # invert the input image (make a copy first)
        old = image; image = iulib.bytearray(); image.copy(old)
        iulib.sub(255,image)

        # allocate working arrays
        segs = iulib.intarray()
        raw = iulib.bytearray()
        mask = iulib.bytearray()

        # now iterate through the characters and collect candidates
        inputs = []
        for i in range(self.grouper.length()):
            # get the bounding box for the character (used later)
            bbox = self.grouper.boundingBox(i)
            aspect = bbox.height()*1.0/bbox.width()

            # extract the character image (and optionally display it)
            self.grouper.extractWithMask(raw,mask,image,i,1)
            char = NI(raw)
            char = char / float(amax(char))
            if self.debug:
                subplot(4,1,2)
                print i,(bbox.x0,bbox.y0,bbox.x1,bbox.y1)
                cla()
                imshow(char,cmap=cm.gray)
                draw()
                print "hit RETURN to continue"
                raw_input()
            inputs.append(FI(char))

        # classify the candidates (using multithreading, where available)
        results = utils.omp_classify(self.cmodel,inputs)
        
        # now convert the classified outputs into a list of candidate records
        candidates = []
        for i in range(len(inputs)):
            # compute the classifier output for this character
            # print self.cmodel.info()
            raw = inputs[i]
            char = NI(raw)
            bbox = self.grouper.boundingBox(i)
            outputs = results[i]
            outputs = [(x[0],-log(x[1])) for x in outputs]
            candidates.append(Record(index=i,image=char,raw=raw,outputs=outputs,bbox=bbox))

        # keep the characters around for debugging (used by ocropus-showlrecs)
        self.chars = candidates

        # update the per-character costs based on a text line model
        if self.linemodel is not None:
            self.linemodel.linecosts(candidates,image)

        # compute a list of space costs for each candidate character
        spacecosts = self.spacemodel.spacecosts(candidates,image)
        
        for c in candidates:
            i = c.index
            raw = c.raw
            char = c.image
            outputs = c.outputs

            # Add a skip transition with the pixel width as cost.
            # This ensures that the lattice is at least connected.
            # Note that for typical character widths, this is going
            # to be much larger than any per-charcter cost.
            self.grouper.setClass(i,ocropus.L_RHO,self.rho_scale*raw.dim(0))
            
            # add the top classes to the lattice
            outputs.sort(key=lambda x:x[1])
            s = iulib.ustrg()
            for cls,cost in outputs[:self.best]:
                # don't add the reject class (written as "~")
                if cls=="~": continue

                # don't add anything with a cost higher than the reject cost
                if cost>self.reject_cost: continue

                # for anything else, just add the classified character to the grouper
                s = iulib.unicode2ustrg(cls)
                self.grouper.setClass(i,s,min(cost,self.maxcost))

                # add the computed space costs to the grouper as well
                self.grouper.setSpaceCost(i,spacecosts[i][0],spacecosts[i][1])

        # extract the recognition lattice from the grouper
        self.grouper.getLattice(lattice)

        # return the raw segmentation as a result
        return rseg

    def startTraining(self,type="adaptation",model="LatinClassifier"):
        """Instantiate a new character model, plus size and space models."""
        self.new_model = ocropy.make_IModel(model)
        self.new_model.pset("cds","bitdataset")
        self.new_model.setExtractor("StandardExtractor")

    def finishTraining(self):
        """After all the data has been loaded, this performs the actual training."""
        self.new_model.updateModel()
        self.cmodel = self.new_model
        self.new_model = None

    def addTrainingLine1(self,image,transcription):
        """Add a line of text plus its transcription to the line recognizer as
        training data."""
        self.addTrainingLine(ocropy.intarray(),image,transcription)

    def addTrainingLine(self,rseg,image_,transcription):
        """Add a line of text plus its transcription to the line recognizer as
        training data. This also returns the raw segmentation in its first
        argument (an intarray)."""
        # rseg = ocropy.intarray()

        # make a copy of the input image
        image = ocropy.bytearray()
        image.copy(image_)

        # now run the recognizer
        lattice = ocropy.make_OcroFST()
        self.recognizeLineSeg(lattice,rseg,image)
        print "bestpath",bestpath(lattice)

        # compute the alignment
        print "gt",transcription
        lmodel = fstutils.make_line_fst([transcription])
        r = ocropy.compute_alignment(lattice,rseg,lmodel)
        result = r.output
        cseg = r.cseg
        costs = r.costs.numpy()
        tcost = sum(costs)
        if tcost>10000.0: raise Exception("cost too high")
        mcost = mean(costs)
        if mcost>10.0: raise RejectException("mean cost too high")
        if tcost>100.0: raise RejectException("total cost too high")
        print "alignment",mcost,tcost

        # this is a special case of ocropus-extract-csegs...

        # find all the aligned characters
        ocropy.sub(255,image)
        utext = ocropy.ustrg()
        # utext.assign(text)
        utext.assign(r.output)
        self.grouper.setSegmentationAndGt(rseg,cseg,utext)
        chars = []
        for i in range(self.grouper.length()):
            cls = self.grouper.getGtClass(i)
            if cls==-1: continue # ignore missegmented characters (handled separately below)
            cls = chr(cls)
            raw = ocropy.bytearray()
            mask = ocropy.bytearray()
            self.grouper.extractWithMask(raw,mask,image,i,1)
            chars.append(Record(raw=raw,mask=mask,cls=cls,index=i,bbox=self.grouper.boundingBox(i)))

        # find all the non-aligned groups and add them as nonchars
        bboxes = [rect(c.bbox) for c in chars]
        self.grouper.setSegmentation(rseg)
        nonchars = []
        for i in range(self.grouper.length()):
            bbox = self.grouper.boundingBox(i)
            fractions = [min(bbox.fraction_covered_by(c.bbox),c.bbox.fraction_covered_by(bbox)) for c in chars]
            covered = max(fractions)
            assert covered>1e-5
            if covered>self.maxoverlap: continue
            assert rect(bbox) not in bboxes
            raw = ocropy.bytearray()
            mask = ocropy.bytearray()
            self.grouper.extractWithMask(raw,mask,image,i,1)
            nonchars.append(Record(raw=raw,mask=mask,cls=cls,index=i,bbox=self.grouper.boundingBox(i)))

        # finally add them to the character model
        for c in chars:
            self.new_model.cadd(floatimage(c.raw),c.cls)
        for c in nonchars:
            self.new_model.cadd(floatimage(c.raw),"~")

        self.chars = chars
        self.nonchars = nonchars
        print "#chars",len(chars),"#nonchars",len(nonchars)

    def align(self,chars,cseg,costs,image,transcription):
        """Align an image with its transcription, returning the characters, cseg,
        and per-character costs."""
        lattice = ocropy.make_OcroFST()
        self.recognizeLineSeg(lattice,rseg,image)
        print "bestpath",bestpath(lattice)
        lmodel = self.makeLineModel(transcription)
        r = ocropy.compute_alignment(lattice,rseg,lmodel)
        result = r.output
        costs.copy(r.costs)
        chars.clear() # FIXME
        raise Exception("unimplemented")

    def makeLineModel(self,s):
        raise Exception("unimplemented")

    def epoch(self,n):
        """For stochastic gradient descent, models are often trained for multiple
        epochs.  Method is used to notify this class that a new training epoch has
        started. It is just passed on to the model."""
        if hasattr(self.cmodel,"epoch"):
            self.cmodel.epoch(n)

    def save(self,file):
        """Save the line recognizer as a .pymodel (the file name must
        end in .pymodel)."""
        assert ".pymodel" in file
        with open(file,"w") as stream:
            pickle.dump(self,stream)

    def load(self,file):
        """Load a line recognizer.  This handles both a .cmodel and a .pymodel
        file."""
        self.set_defaults()
        if "+" in file:
            files = file.split("+")
        else:
            files = [file]
        for file in files:
            if ".pymodel" in file:
                with open(file,"r") as stream:
                    obj = cPickle.load(stream)
                if type(obj)==LineRecognizer:
                    for k,v in obj.__dict__:
                        self.__dict__[k] = v
                else:
                    self.cmodel = obj
            elif ".cmodel" in file:
                self.cmodel = ocropy.load_IModel(file)
            elif ".csize" in file:
                self.linemodel = SimpleLineModel()
                self.linemodel.load(file)
            else:
                raise Exception("unknown extension")
