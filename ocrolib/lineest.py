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
    cy,cx = measurements.center_of_mass(image)
    return (sum(((arange(len(image))-cy)[:,newaxis]*image)**2)/sum(image))**.5,cy,cx

def extract_chars(segmentation,h=32,w=32,f=0.5):
    ls,ly,lx = vertical_stddev(segmentation>0)
    boxes = measurements.find_objects(segmentation)
    for i,b in enumerate(boxes):
        sub = (segmentation==i+1)
        cs,cy,cx = vertical_stddev(sub)
        scale = f*h/(4*max(cs,0.1*ls))
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



def build_shape_dictionary(fnames,k=1024,d=0.9):
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



def compute_geomaps(fnames,shapedict,use_gt=1,size=32):
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
        try:
            xh,bl = lineproc.estimate_xheight(image)
        except:
            traceback.print_exc()
            continue
        blimage = zeros(image.shape)
        blimage[bl,:] = 1
        xlimage = zeros(image.shape)
        xlimage[bl-xh,:] = 1
        try: seg = lineseg.ccslineseg(image)
        except: continue
        shape = None
        for sub,transform,itransform_add in extract_chars(seg):
            if shape is None: shape = sub.shape
            assert sub.shape==shape
            count += 1
            best = shapedict.predict1(sub)
            bls[best] += transform(blimage)
            xls[best] += transform(xlimage)
    for i in range(len(bls)): bls[i] *= bls[i].shape[1]*1.0/max(1e-6,sum(bls[i]))
    for i in range(len(xls)): xls[i] *= xls[i].shape[1]*1.0/max(1e-6,sum(xls[i]))
    return bls,xls

if 0:
    ocrolib.showgrid(shapedict.centers().reshape(*xls.shape)+xls*2)
    ocrolib.showgrid(shapedict.centers().reshape(*xls.shape)+bls*2)



def blxlimages(image,shapedict,bls,xls):
    image = (image>ocrolib.midrange(image))
    seg = lineseg.ccslineseg(image)
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
    return params



class LineEstimationModel:
    def __init__(self,k):
        self.k = k
    def buildShapeDictionary(self,fnames):
        self.shapedict = build_shape_dictionary(fnames,k=k)
    def buildGeomaps(self,fnames):
        self.bls,self.xls = compute_geomaps(fnames,self.shapedict)
    def lineFit(self,image,order=1):
        blimage,xlimage = blxlimages(image,self.shapedict,self.bls,self.xls)
        self.blimage = blimage # for debugging
        self.xlimage = xlimage
        blp = fit_peaks(blimage,order=order)
        xlp = fit_peaks(xlimage,order=order)
        return blp,xlp
    def lineParameters(self,image):
        blp,xlp = self.lineFit(image)
        xs = range(image.shape[1])
        bl = mean(polyval(blp,xs))
        xl = mean(polyval(xlp,xs))
        return bl,bl-xl



def expand(fname):
    if fname[0]=="@":
        return open(fname[1:]).read().split("\n")
    elif "?" in fname or "*" in fname:
        return sorted(glob.glob(fname))
    else:
        raise Exception("argument must either be a glob pattern or start with an '@'")


if __name__=="__main__":
    parser= argparse.ArgumentParser("Training for line estimation models.")
    subparsers = parser.add_subparsers(dest="subcommand")

    ptrain = subparsers.add_parser("train")
    parser.add_argument("-e","--estimator",default=None,dest='em_estimator',
                        help="starting estimator for EM step")
    ptrain.add_argument("-L","--dictlines",default="book/????/??????.png",required=1,
                        help="list of text lines to be used for building the shape dictionary")
    ptrain.add_argument("-G","--geolines",default="book/????/??????.png",required=1,
                        help="list of text lines to be used for building the geometric models")
    ptrain.add_argument("-k","--nprotos",type=int,default=1024)
    ptrain.add_argument("-o","--output",default="default.lineest",required=1,
                        help="output file for the pickled line estimator")
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
    pshowline.add_argument("image",default=None)

    args = parser.parse_args()
    if args.subcommand=="train":
        lem = LineEstimationModel(k=args.nprotos)
        lem.buildShapeDictionary(expand(args.dictlines))
        lem.buildGeomaps(expand(args.geolines))
        with open(args.output,"w") as stream:
            cPickle.dump(lem,stream)
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
        image = 1-ocrolib.read_image_gray(args.image)
        limit = min(image.shape[1],args.xlimit)
        blp,xlp = lem.lineFit(image,order=args.order)
        print lem.lineParameters(image)
        subplot(211); imshow((lem.blimage-lem.xlimage)[:,:limit])
        gray()
        subplot(212); imshow(image[:,:limit])
        xlim(0,limit); ylim(len(image),0)
        xs = range(image.shape[1])[:limit]
        plot(xs,polyval(blp,xs))
        plot(xs,polyval(xlp,xs))
        ginput(1,1000)
        sys.exit(0)

