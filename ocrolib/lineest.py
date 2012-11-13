import sys,os,re,glob,math,glob,signal
import scipy
from scipy import stats
from scipy.ndimage import measurements,interpolation,morphology,filters
from pylab import *
import common,sl,morph
import ocrolib
from ocrolib import lineseg,lineproc
from toplevel import *
import re
import glob
from ocrolib import patrec
import argparse
import cPickle



def vertical_stddev(image):
    """Compute the standard deviation in the vertical direction.
    This is used below to get a rough idea of how large characters
    and lines are."""
    assert amax(image)>amin(image)
    cy,cx = measurements.center_of_mass(1.0*image)
    return (sum(((arange(len(image))-cy)[:,newaxis]*image)**2)/sum(image))**.5,cy,cx

def extract_chars(segmentation,h=32,w=32,f=0.5,minscale=0.5):
    """Extract all the characters from the segmentation and yields them
    as an interator.  Also yields a forward and a backwards transformation."""
    bin = (segmentation>0)
    if amax(bin)==0: raise ocrolib.RecognitionError("empty segmentation")
    ls,ly,lx = vertical_stddev(bin)
    boxes = morph.find_objects(segmentation)
    for i,b in enumerate(boxes):
        sub = (segmentation==i+1)
        if amax(sub)==amin(sub): continue
        cs,cy,cx = vertical_stddev(sub)
        # limit the character sigma to be at least minscale times the
        # line sigma (this causes dots etc. not to blow up ridiculously large)
        scale = f*h/(4*max(cs,minscale*ls))
        m = diag([1.0/scale,1.0/scale])
        offset = array([cy,cx])-dot(m,array([h/2,w/2]))
        def transform(image,m=m,offset=offset):
            return interpolation.affine_transform(1.0*image,m,offset=offset,order=1,output_shape=(h,w))
        def itransform_add(result,image,m=m,cx=cx,cy=cy):
            im = inv(m)
            ioffset = array([h/2,w/2])-dot(im,array([cy,cx]))
            result += interpolation.affine_transform(1.0*image,im,offset=ioffset,order=1,output_shape=segmentation.shape)
        cimage = transform(sub)
        yield cimage,transform,itransform_add



def build_shape_dictionary(fnames,k=1024,d=0.9,debug=0):
    """Given a collection of file names, create a shape dictionary."""
    def allchars():
        count = 0
        for fno,fname in enumerate(fnames):
            if fno%20==0: print fno,fname,count
            image = 1-ocrolib.read_image_gray(fname)
            try:
                seg = lineseg.ccslineseg(image)
            except:
                traceback.print_exc()
                continue
            seg = morph.renumber_by_xcenter(seg)
            for e in extract_chars(seg):
                count += 1
                yield e
    data = [x for x,_,_ in allchars()]
    data = array(data,'float32').reshape(len(data),-1)
    km = patrec.PcaKmeans(k,d,verbose=1)
    km.fit(data)
    return km



def compute_geomaps(fnames,shapedict,old_model,use_gt=1,size=32,debug=0,old_order=1):
    """Given a shape dictionary and an existing line geometry
    estimator, compute updated geometric maps for each entry
    in the shape dictionary."""
    if debug>0: gray(); ion()
    shape = (shapedict.k,size,size)
    bls = zeros(shape)
    xls = zeros(shape)
    count = 0
    for fno,fname in enumerate(fnames):
        if fno%20==0: print fno,fname,count
        if use_gt:
            # don't use lines with many capital letters for training because
            # they result in bad models
            gt = ocrolib.read_text(ocrolib.fvariant(fname,"txt","gt"))
            if len(re.sub(r'[^A-Z]','',gt))>=0.3*len(re.sub(r'[^a-z]','',gt)): continue
            if len(re.sub(r'[^0-9]','',gt))>=0.3*len(re.sub(r'[^a-z]','',gt)): continue
        image = 1-ocrolib.read_image_gray(fname)
        if debug>0 and fno%debug==0: clf(); subplot(411); imshow(image)
        try:
            blp,xlp = old_model.lineFit(image,order=old_order)
        except:
            traceback.print_exc()
            continue
        blimage = zeros(image.shape)
        h,w = image.shape
        for x in range(w): blimage[clip(int(polyval(blp,x)),0,h-1),x] = 1
        xlimage = zeros(image.shape)
        for x in range(w): xlimage[clip(int(polyval(xlp,x)),0,h-1),x] = 1
        if debug>0 and fno%debug==0: 
            subplot(413); imshow(xlimage+0.3*image)
            subplot(414); imshow(blimage+0.3*image)
        try: 
            seg = lineseg.ccslineseg(image)
        except: 
            continue
        if debug>0 and fno%debug==0: subplot(412); morph.showlabels(seg)
        shape = None
        for sub,transform,itransform_add in extract_chars(seg):
            if shape is None: shape = sub.shape
            assert sub.shape==shape
            count += 1
            best = shapedict.predict1(sub)
            bls[best] += transform(blimage)
            xls[best] += transform(xlimage)
        if debug==1: ginput(1,100)
        elif debug>1: ginput(1,0.01)
    for i in range(len(bls)): bls[i] *= bls[i].shape[1]*1.0/max(1e-6,sum(bls[i]))
    for i in range(len(xls)): xls[i] *= xls[i].shape[1]*1.0/max(1e-6,sum(xls[i]))
    return bls,xls

if 0:
    ocrolib.showgrid(shapedict.centers().reshape(*xls.shape)+xls*2)
    ocrolib.showgrid(shapedict.centers().reshape(*xls.shape)+bls*2)



def blxlimages(image,shapedict,bls,xls):
    image = (image>ocrolib.midrange(image))
    if amax(image)==0: raise RecognitionError("empty line")
    seg = lineseg.ccslineseg(image)
    # ion(); subplot(311); imshow(image); subplot(312); morph.showlabels(seg); ginput(1,0.1); raw_input()
    seg = morph.renumber_by_xcenter(seg)
    blimage = zeros(image.shape)
    xlimage = zeros(image.shape)
    for sub,transform,itransform_add in extract_chars(seg):
        best = shapedict.predict1(sub)
        bli = bls[best].reshape(32,32)
        xli = xls[best].reshape(32,32)
        itransform_add(blimage,bli)
        itransform_add(xlimage,xli)
    return blimage,xlimage

def fit_peaks(smoothed,order=1,filter_size=(1.0,20.0)):
    """TODO: use robust fitting to deal with multiple peaks"""
    smoothed = filters.gaussian_filter(smoothed,filter_size)
    ys = argmax(smoothed,axis=0)
    params = polyfit(arange(len(ys)),ys,order)
    # print "fit_peaks"
    # clf(); imshow(smoothed); plot(arange(len(ys)),ys); ginput(1,99)
    return params



class GradientLineGeometry:
    def lineFit(self,image,order=1):
        """Assuming the line is horizontal, compute xheight and baseline."""
        xh,bl = lineproc.estimate_xheight(image)
        return array([bl]),array([bl-xh])
    def lineParameters(self,image,order=1):
        """Assuming the line is horizontal, compute xheight and baseline, and all
        the other parameters."""
        xh,bl = lineproc.estimate_xheight(image)
        return bl,bl-xh,array([bl]),array([bl-xh])

class TrainedLineGeometry:
    def __init__(self,k=256,d=0.9):
        self.k = k
        self.d = d
    def buildShapeDictionary(self,fnames,debug=0):
        """Build a shape dictionary from a list of text line files."""
        self.shapedict = build_shape_dictionary(fnames,k=self.k,d=self.d,debug=debug)
    def buildGeomaps(self,fnames,old_model=GradientLineGeometry(),debug=0,old_order=1):
        """Build geometric maps from a list of text line files."""
        self.bls,self.xls = compute_geomaps(fnames,self.shapedict,old_model=old_model,old_order=old_order,debug=debug)
    def lineFit(self,image,order=1):
        """Return polynomial fits to the baseline and xline."""
        blimage,xlimage = blxlimages(image,self.shapedict,self.bls,self.xls)
        self.blimage = blimage # for debugging
        self.xlimage = xlimage
        #clf(); imshow(blimage); title("blimage"); ginput(1,99)
        #clf(); imshow(xlimage); title("xlimage"); ginput(1,99)
        blp = fit_peaks(blimage,order=order)
        xlp = fit_peaks(xlimage,order=order)
        return blp,xlp
    def lineParameters(self,image,order=1):
        """Return the average baseline and average xheight,
        plus the polynomial models for both"""
        blp,xlp = self.lineFit(image,order=order)
        xs = range(image.shape[1])
        bl = mean(polyval(blp,xs))
        xl = mean(polyval(xlp,xs))
        return bl,bl-xl,blp,xlp

# older name, useful for unpickling old versions

LineEstimationModel = TrainedLineGeometry



# General purpose functions for line dewarping.

def force_height(image,target_h,cval=0):
    """Force a line to the given target height by cropping
    vertically or adding background pixels."""
    h,w = image.shape
    result = ones((target_h,w),dtype=image.dtype)*cval
    h = min(h,target_h)
    result[:h,:] = image[:h,:]
    return result

def make_normalizer(emodel,limage,target_height,params=(0.5,0.8,1)):
    xheightfrac,baselinefrac,degree = params
    target_xheight = int(0.5+xheightfrac*target_height)
    target_baseline = int(0.5+baselinefrac*target_height)
    return make_normalizer_abs(emodel,limage,target_height,target_xheight,target_baseline,degree=degree)

def make_normalizer_abs(emodel,limage,target_height,target_xheight,target_baseline,degree=1):
    """Given an emodel, a line image, and targets for xheight and baselines,
    this returns a function that normalizes images of the same size and shape
    as limage. This can be used to normalize the original line image,
    as well as cseg or other line-related images."""
    rimage = 1.0-limage*1.0/amax(limage)
    avgbaseline,xheight,blp,xlp = emodel.lineParameters(rimage,order=degree)
    h,w = limage.shape
    scale = (1+target_xheight)*1.0/xheight
    missing = target_height/scale-h
    bs = target_baseline/scale
    def normalize(img,order=1,dtype=dtype('f'),cval=0):
        img = img.copy()
        assert img.shape==(h,w)
        if missing>0:
            img = vstack([img,ones([missing,w],dtype=dtype)*cval])
        for x in range(w):
            baseline = polyval(blp,x)
            img[:,x] = interpolation.shift(img[:,x],bs-baseline,order=order,mode='constant',cval=cval,output=dtype)
        img = interpolation.zoom(img,scale,order=1)
        img = force_height(img,target_height,cval=cval)
        return img
    return normalize



default_params = (0.5,0.8,1)

class LineestNormalizer:
    def __init__(self,emodel,target_height=32,params=default_params):
        self.emodel = emodel
        self.target_height = target_height
        self.params = params
    def setHeight(self,target_height):
        self.target_height = target_height
    def measure(self,limage):
        self.normalize = make_normalizer(self.emodel,amax(limage)-limage,self.target_height,self.params)
    def normalize(self,img,*args,**kw):
        return self.normalize(img,*args,**kw)

class MvNormalizer:
    def __init__(self,target_height=32):
        self.target_height = target_height
        self.perc = 90
        self.dest = 0.4
        self.sigma = (10,30)
    def setHeight(self,h):
        self.target_height = h
    def measure(self,image):
        global xs,center
        self.shape = image.shape
        h,w = image.shape
        ys = arange(h)
        xs = arange(w)
        # smooth the image and compute the vertical mean
        smoothed = filters.gaussian_filter(image,self.sigma)
        center = sum(smoothed*ys[:,newaxis],axis=0)/sum(1e-6+smoothed,axis=0)
        # perform a linear fit
        p = polyfit(xs,center,1)
        m,b = p
        # compute the deviation of the image pixels from the line (m,b)
        yps = (ys[:,newaxis]-polyval(p,xs)[newaxis,:])
        yps = abs(yps)[image>0.5]
        d = stats.scoreatpercentile(yps,self.perc)
        # we want this distance to be a fraction of dest of the target height;
        # store the appropriate normalization parameters
        self.scale = d/((self.target_height/2)*self.dest)
        self.offset = b-self.scale*self.target_height/2
        self.m = m
        #print self.scale,self.offset,self.m
    def normalize(self,img,order=1,dtype=dtype('f'),cval=0):
        assert img.shape==self.shape
        h,w = self.shape
        M = array([[self.scale,self.scale*self.m],[0,self.scale]])
        offset = (self.offset,0)
        shape = (self.target_height,int(w/self.scale)) 
        return interpolation.affine_transform(img,M,offset=offset,output_shape=shape,order=order,cval=cval,output=dtype)

def load_normalizer(fname):
    fname = ocrolib.findfile(fname)
    model = ocrolib.load_component(fname)
    if "lineParameters" in dir(model):
        return LineestNormalizer(model,params=default_params)
    if "normalize" in dir(model):
        return model
    raise Exception("model seems to be neither a normalizer nor a line estimator")



def expand(fname):
    if fname[0]=="@":
        fnames = open(fname[1:]).read().split("\n")
        fnames = [f for f in fnames if f!="" and os.path.exists(f)]
        return fnames
    elif "?" in fname or "*" in fname:
        return sorted(glob.glob(fname))
    else:
        raise Exception("argument must either be a glob pattern or start with an '@'")


if __name__=="__main__":
    parser= argparse.ArgumentParser("Training for line estimation models.")
    subparsers = parser.add_subparsers(dest="subcommand")

    ptrain = subparsers.add_parser("train")
    ptrain.add_argument("-e","--estimator",default=None,dest='em_estimator',
                        help="starting estimator for EM step")
    ptrain.add_argument("-E","--order",default=1,type=int,
                        help="order for polynomial fit of line estimator")
    ptrain.add_argument("-L","--dictlines",default="book/????/??????.png",required=1,
                        help="list of text lines to be used for building the shape dictionary")
    ptrain.add_argument("-G","--geolines",default="book/????/??????.png",required=1,
                        help="list of text lines to be used for building the geometric models")
    ptrain.add_argument("-k","--nprotos",type=int,default=1024)
    ptrain.add_argument("-d","--ndims",type=float,default=0.95)
    ptrain.add_argument("-o","--output",default="default.lineest",required=1,
                        help="output file for the pickled line estimator")
    ptrain.add_argument("--debug",type=int,default=0)
    # FIXME There are a lot more training parameters that could be exposed here.

    pshowdict = subparsers.add_parser("showdict")
    pshowdict.add_argument("-e","--estimator",default=None,dest="show_estimator",required=1,
                           help="estimator to be displayed")

    pshowline = subparsers.add_parser("showline")
    pshowline.add_argument("-e","--estimator",default=None,dest="line_estimator",required=1,
                           help="estimator to be used")
    pshowline.add_argument("-x","--xlimit",type=int,default=2000,
                           help="only display the left part of the line up to this point for better visibility")
    pshowline.add_argument("-p","--order",type=int,default=1)
    pshowline.add_argument("images",nargs='+',default=[])

    args = parser.parse_args()
    if args.subcommand=="train":
        # apparently, we need this to make pickling work out correctly
        import ocrolib.lineest
        estimator = ocrolib.lineest.GradientLineGeometry()
        if args.em_estimator is not None:
            with open(args.em_estimator) as stream:
                estimator = cPickle.load(stream)
        lem = TrainedLineGeometry(k=args.nprotos,d=args.ndims)
        lem.buildShapeDictionary(expand(args.dictlines))
        lem.buildGeomaps(expand(args.geolines),
            old_model=estimator,old_order=args.order,
            debug=args.debug)
        with open(args.output,"w") as stream:
            cPickle.dump(lem,stream,2)
        sys.exit(0)
    elif args.subcommand=="showdict":
        with open(args.show_estimator) as stream:
            lem = cPickle.load(stream)
        print "loaded",lem
        km = lem.shapedict
        ion(); gray()
        ocrolib.showgrid(km.centers().reshape(*lem.xls.shape)+lem.xls*2)
        ginput(1,1000)
        ocrolib.showgrid(km.centers().reshape(*lem.bls.shape)+lem.bls*2)
        ginput(1,1000)
        sys.exit(0)
    elif args.subcommand=="showline":
        with open(args.line_estimator) as stream:
            lem = cPickle.load(stream)
        print "loaded",lem
        for fname in args.images:
            try:
                print "***",fname
                clf()
                image = 1-ocrolib.read_image_gray(fname)
                limit = min(image.shape[1],args.xlimit)
                blp,xlp = lem.lineFit(image,order=args.order)
                print "baseline",blp
                print "xline",xlp
                title("fname")
                subplot(311); imshow((lem.blimage-lem.xlimage)[:,:limit])
                title("fname")
                subplot(312); imshow((lem.blimage-lem.xlimage+image)[:,:limit])
                gray()
                subplot(313); imshow(image[:,:limit])
                xlim(0,limit); ylim(len(image),0)
                xs = range(image.shape[1])[:limit]
                plot(xs,polyval(blp,xs))
                plot(xs,polyval(xlp,xs))
                ginput(1,1000)
            except:
                print "ERROR IN IMAGE",fname
                traceback.print_exc()
                continue
        sys.exit(0)

