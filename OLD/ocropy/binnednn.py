import sys,os,re,glob,math,glob,numpy
import iulib,ocropus
from pylab import *
from scipy.ndimage import measurements,interpolation
from scipy.ndimage.morphology import binary_erosion,binary_dilation,binary_opening,binary_closing
import dbtables

# bin statistics, including distances etc.
# make allowable variation relative to stroke width?

# automatic tuning by comparing with ground truth classes
# -- set parameters so that class confusions are rare

# tree vq, hierarchical vq
# bitflip cost rather than squared difference
# margin flip cost rather than spatially uniform
# mutually contained skeletons (maybe with a little dilation)

N = iulib.numpy
def NI(image): return transpose(N(image))[::-1,...]
def BA(image):
    result = iulib.bytearray()
    iulib.narray_of_numpy(result,image)
    return result

def square(image):
    w,h = image.shape
    r = max(w,h)
    output = zeros((r,r),image.dtype)
    dx = (r-w)/2
    dy = (r-h)/2
    output[dx:dx+w,dy:dy+h] = image
    return output

def stdsize(image,r=30):
    image = square(image)
    s,_ = image.shape
    return interpolation.zoom(1.0*image,(r+0.5)/float(s),order=3)

def skeletal_feature_counts(image,presmooth=0.0,skelsmooth=0.0):
    image = array(255*(image>0.5),'B')
    image = BA(image)
    val = ocropus.skeletal_feature_hack(image,presmooth,skelsmooth)
    return (val/1000,val%1000)

def thin(image,c=1):
    if c>0: image = binary_closing(image,iterations=c)
    image = array(image,'B')
    image = BA(image)
    iulib.thin(image)
    return array(N(image),'B')

def pad_to(image,w,h):
    iw,ih = image.shape
    wd = int(w-iw)
    assert wd>=0
    w0 = wd/2
    w1 = wd-w0
    hd = int(h-ih)
    assert hd>=0
    h0 = hd/2
    h1 = hd-h0
    result = zeros((w,h))
    result[w0:w0+iw,h0:h0+ih] = image
    return result

def pad_by(image,w,h):
    iw,ih = image.shape
    return pad_to(image,iw+2*w,ih+2*h)

def pad_bin(char,r=5):
    w,h = char.shape
    w = r*int((w+r-1)/r)
    h = r*int((h+r-1)/r)
    return pad_to(char,w,h)

def make_mask(image,r):    
    skeleton = thin(image)
    mask = ~(binary_dilation(image,iterations=r) - binary_erosion(image,iterations=r))
    mask |= skeleton # binary_dilation(skeleton,iterations=1)
    return mask

def dist(image,item):
    assert image.shape==item.shape,[image.shape,item.shape]
    ix,iy = measurements.center_of_mass(image)
    if isnan(ix) or isnan(iy): return 9999,9999,None
    # item = (item>amax(item)/2) # in case it's grayscale
    x,y = measurements.center_of_mass(item)
    if isnan(x) or isnan(y): return 9999,9999,None
    dx,dy = int(0.5+x-ix),int(0.5+y-iy)
    shifted = interpolation.shift(image,(dy,dx))
    if abs(dx)>2 or abs(dy)>2:
        return 9999,9999,None
    if 0:
        cla()
        subplot(121); imshow(image-item)
        subplot(122); imshow(shifted-item)
        show()
    image = shifted
    mask = make_mask(image>0.5,1)
    err = sum(mask*abs(item-image))
    total = min(sum(mask*item),sum(mask*image))
    rerr = err/max(1.0,total)
    return err,rerr,image

def symdist(image,item):
    assert type(image)==numpy.ndarray
    assert len(image.shape)==2
    assert len(item.shape)==2
    err,rerr,transformed = dist(image,item)
    err1,rerr1,transformed1 = dist(item,image)
    if rerr<rerr1: return err,rerr,transformed
    else: return err1,rerr1,transformed1

def remerge(self,verbose=0):
    for key,bin in self.bins.items():
        i = 0
        while i<len(bin):
            if bin[i] is None: continue
            j = i+1
            while j<len(bin):
                if bin[j] is None: continue
                err,rerr,image = symdist(bin[j].image,bin[i].image)
                if err<self.eps or rerr<self.reps:
                    if verbose:
                        print "merging",key,i,j,"err",err,"rerr",rerr
                    # fixme really need to use image here
                    bin[i].merge(bin[j])
                    del bin[j]
                j += 1
            i += 1

def hashappend(h,k,e):
    h[k] = h.get(k,[])+[e]
def hashinc(h,k,inc=1):
    h[k] = h.get(k,0)+inc

pattern_id = 1

class Pattern:
    def __init__(self,a=None,cls=None):
        global pattern_id
        self.id = pattern_id
        pattern_id += 1
        if a is None:
            self.image = None
            self.count = 0
            self.classes = {}
        else:
            self.image = a
            self.count = 1
            self.classes = {cls:1}
    def dist(self,other):
        return symdist(self.image,other.image)
    def add(self,a,cls=None):
        hashinc(self.classes,cls)
        if self.image is None:
            self.image = a
            self.count = 1
        else:
            self.image = (self.count*self.image+a) * 1.0/(self.count+1)
            self.count += 1
    def merge(self,pat):
        self.image = (float(self.count)*self.image + float(pat.count)*pat.image) / \
            (float(self.count)+float(pat.count))
        self.count += pat.count
        for k,v in pat.classes.items(): hashinc(self.classes,k,v)
    def cls(self):
        l = sorted([(v,k) for k,v in self.classes.items()],reverse=1)
        return l[0][1]

class BinnedNN:
    def __init__(self,eps=7,reps=0.07):
        self.count = 0
        self.eps = eps
        self.reps = reps
        self.clear()
        self.hole_bins = 1
        self.skel_bins = 0
        self.fixed = 1 # stop as soon as the boundaries are matched
        if self.fixed: print "[binnednn fixed mode]"
        self.sort_freq = 0.01   # how often to sort the prototypes
    def clear(self):
        self.bins = {}
    def biniter(self,mincount=0):
        for key in self.bins.keys():
            bin = self.bins[key]
            for i in range(len(self.bins[key])):
                pat = bin[i]
                if pat.count<mincount: continue
                yield key,i,pat.count,pat
    def collect(self,maxage,maxcount=1):
        collected = 0
        for key in self.bins.keys():
            bin = self.bins[key]
            for i in range(len(self.bins[key])-1,-1,-1):
                pat = bin[i]
                if self.count-pat.when>maxage and pat.count<=maxcount:
                    del bin[i]
                    collected += 1
                    continue
        return collected
    def load(self,file):
        table = dbtables.Table(file,"clusters")
        table.converter("image",dbtables.SmallImage())
        for row in table.get():
            key = eval(row.key,{},{}) # not secure...
            pat = Pattern()
            pat.image = row.image/float(255)
            pat.count = row.count
            pat.cls = row.cls
            pat.classes = eval(row.classes,{},{})
            bin = self.bins.get(key,[])
            bin.append(pat)
            self.bins[key] = bin
    def save(self,file):
        table = dbtables.ClusterTable(file)
        table.create(image="blob",count="integer",cls="text",classes="text",key="text")
        for key,i,count,pat in self.biniter():
            table.set(id=pat.id,image=pat.image,cls=pat.cls(),
                      count=pat.count,classes=str(pat.classes),key=str(key))
    def stats(self):
        bins = 0
        total = 0
        for key,i,count,image in self.biniter():
            total += count
            bins += 1
        return bins,total
    def find(self,image,key):
        items = self.bins.get(key,[])
        if random()<self.sort_freq:
            items.sort(key=lambda item:item.count,reverse=1)
            self.bins[key] = items
        if not self.fixed:
            best_i = -1
            best_err = 1e38
            best_rerr = 1e38
            best_transformed = None
            for i in range(len(items)):
                item = items[i]
                err,rerr,transformed = dist(image,item.image)
                if err>=best_err: continue
                best_err = err
                best_rerr = rerr
                best_transformed = transformed
                best_i = i
            if best_i>=0:
                return (best_err,best_rerr,(key,best_i),best_transformed)
            else:
                return None
        else:
            for i in range(len(items)):
                item = items[i]
                err,rerr,transformed = dist(image,item.image)
                if err<self.eps or rerr<self.reps:
                    return (err,rerr,(key,i),transformed)
            return None
    def addKeyed(self,image,xkey,cls=None):
        key = tuple(list(image.shape)+list(xkey))
        result = self.find(image,key)
        if result is None:
            pat = Pattern(image,cls)
            pat.when = self.count
            hashappend(self.bins,key,pat)
        else:
            err,rerr,(key,i),transformed = result
            if err>self.eps and rerr>self.reps:
                pat = Pattern(image,cls)
                pat.when = self.count
                hashappend(self.bins,key,pat)
            else:
                pat = self.bins[key][i]
                pat.add(transformed,cls)
                pat.when = self.count
        self.count += 1
    def descriptor(self,char):
        xkey = list(char.shape)
        if self.hole_bins:
            _,nholes = measurements.label(1-binary_closing(pad_by(char,2,2)))
            nholes -= 1
            xkey += [nholes]
        if self.skel_bins:
            xkey = xkey + list(skeletal_feature_counts(char,1.0,1.0))
        return tuple(xkey)
    def add(self,char,cls=None):
        char = pad_bin(char)
        xkey = self.descriptor(char)
        self.addKeyed(char,xkey,cls=cls)
    def nearest(self,char):
        char = pad_bin(char)
        key = self.descriptor(char)
        result = self.find(char,key)
        if result is None: return None
        err,rerr,(key,i),transformed = result
        return self.bins[key][i],err,rerr
    def neighbors(self,char):
        char = pad_bin(char)
        assert amax(char)<1.1
        key = self.descriptor(char)
        result = self.find(char,key)
        if result is None: return None
        result = []
        for pat in self.bins[key]:
            err,rerr,transformed = symdist(char,pat.image)
            result.append((err,rerr,pat))
        result = sorted(result)
        if 0:
            print result[:4]
            subplot(2,4,1); imshow(char)
            for i in range(min(4,len(result))):
                subplot(2,4,5+i); imshow(result[i][1].image)
            show()
        return result

def binshow(binned,mincount=2):
    ordered = []
    bins = list(binned.biniter(mincount=mincount))
    bin = None
    while len(bins)>0:
        index = 0
        if bin is not None:
            md = 1e38
            mi = -1
            for i in range(len(bins)):
                b = bins[i]
                if b[3].image.shape!=bin[3].image.shape: continue
                d,_,_ = symdist(bin[3].image,b[3].image)
                if d>=md: continue
                md = d
                mi = i
            index = mi
        bin = bins[index]
        del bins[index]
        cla()
        title(str(bin[3].count)+" "+str(bin[3].classes))
        imshow(bin[3].image)
        gray()
        draw()
        print len(ordered),"key",bin[0],"i",bin[1],"count",bin[2]
        ordered.append(bin)

# binned = BinnedNN(eps=7,reps=0.07)

def cluster(files):
    for file in files:
        print "loading",file
        image = imread(file)
        if len(image.shape)==3: image = average(image,axis=2)
        image = 255-255*array(image>0.5,'B')
        labels,n = measurements.label(image)
        print file,"shape",image.shape,"#labels",n,"max",amax(image)
        slices = measurements.find_objects(labels)
        for i in range(len(slices)):
            u,v = slices[i]
            if u.stop-u.start<=10: continue
            if v.stop-v.start<=10: continue
            char = (labels[slices[i]]==(i+1))
            binned.add(char)
        print binned.stats()
    return binned

def test():
    cluster(["scan-print.png"])   
    # cluster(sorted(glob.glob("book/000[0-9].bin.png")))
    # cluster(glob.glob("jstor-samples/j100268-i329986.book/0014.bin.png"))
    print binned.stats()
    print "remerging"
    print remerge(binned)
    print binned.stats()
    print "showing"
    binshow(binned,1)
