import sys,os,random,math
import numpy,pylab,scipy
from ocropy import improc
import ocropy
import nnlib
from numpy import *
from pylab import find
from scipy.ndimage import filters,morphology
import numpy

def rchoose(k,n):
    assert k<=n
    return random.permutation(range(n))[:k]

def showgrid(images,r=None):
    """Show a grid of character images, for debugging."""
    from pylab import clf,gray,draw,imshow,ginput,subplot
    clf()
    gray()
    if r is None:
        r = int(sqrt(len(images))-1e-9)+1
        r = min(20,r)
    for i in range(min(len(images),r*r)):
        subplot(r,r,i+1)
        image = images[i]
        if len(image.shape)==1:
            d = int(sqrt(len(image)))
            image = image.reshape(d,d)
        imshow(image)
    draw()
    ginput(1,timeout=0.1)

def makergb(r,g,b):
    image = array([r,g,b])
    if len(image.shape)==2:
        d = int(sqrt(len(r)))
        image = image.reshape(3,d,d)
    image = image.transpose([1,2,0])
    image /= amax(image)
    return image

def showgrid3(r,g,b,d=None):
    """Show a grid of character images, for debugging."""
    if d is None:
        d = int(sqrt(len(r))-1e-9)+1
        d = min(20,d)
    from pylab import clf,gray,draw,imshow,ginput,subplot
    r = r/max(1e-30,amax(r))
    g = g/max(1e-30,amax(g))
    b = b/max(1e-30,amax(b))
    clf()
    for i in range(min(len(r),d*d)):
        subplot(d,d,i+1)
        imshow(makergb(r[i],g[i],b[i]))
    draw()
    ginput(1,timeout=0.1)

def classhist(items):
    """Given a list of classes, computes a list of tuples consisting of the number of
    counts and the repeated items."""
    counts = {}
    for item in items:
        if not item in counts:
            counts[item] = 1
        else:
            counts[item] += 1
    return sorted([(v,k) for k,v in counts.items()],reverse=1)

def pathstr(path):
    """Give GmmTree paths (lists of integers) as strings."""
    s = ""
    for p in path:
        s += "%3s."%p
    return s

def more(l):
    """Returns the rest of the list, or just the list itself if it contains only
    one element."""
    if len(l)>1: return l[1:]
    else: return l

def select(l,indexes):
    return [l[i] for i in indexes]

default_maxiter = int(os.getenv("maxiter") or 100)

def square(a):
    if a.ndim==2: return a
    r = int(sqrt(a.size))
    a = a.reshape(r,r)
    return a

def spatially_regularize(a,s):
    a = square(a)
    if s<0:
        a = morphology.grey_dilation(a,(-s,-s))
    elif s>0:
        a = filters.convolve(a,(s,s))
    return a

def fractile(s,f):
    s = s.ravel()
    return sorted(s)[int(f*len(s))]

def gmm_em(data,k,maxiter=default_maxiter,verbose=0,mincount=None,maxupdate=4,reg=0.005,sreg=0,rfrac=0.05,rstart=10):
    verbose = 1
    n = len(data)
    m = len(data[0].ravel())
    if mincount is None:
        mincount = max(3,int(sqrt(n/k)))
    print "clustering",(n,m),"data into",k,"vectors","mincount",mincount
    means = array(select(data,rchoose(k,len(data))),'f')
    sigmas = ones((k,m),'f')
    oldmins = -ones(n,'i')
    for iter in range(maxiter):
        if verbose>0: print "[%d]"%iter,; sys.stdout.flush()
        out = zeros(k,'f')
        offsets = -sum(log(sigmas),axis=1)
        # mins = array([nnlib.argmax_ll(means,sigmas,offsets,data[i]) for i in range(len(data))])
        mins = []
        dists = []
        for i in range(len(data)):
            index,d = nnlib.all_argmax_ll(means,sigmas,offsets,data[i].reshape(1,data[i].size))
            mins.append(index[0])
            dists.append(d[0])
        mins = array(mins,'i')
        dists = array(dists)
        if verbose>1: print mins
        if sum(mins!=oldmins)<maxupdate: break
        reinit = 0
        counts = []
        outliers = 0
        for i in range(k):
            where = (mins==i)
            c = sum(where)
            if iter>=rstart and sum(c)>=20:
                threshold = fractile(dists[where],rfrac)
                # print threshold,amin(dists[where]),amax(dists[where])
                where = (mins==i) * (dists>=threshold)
                old = c
                c = sum(where)
                outliers += old-c
            if sum(c)<mincount:
                where = array([0]*len(data))
                where[rchoose(5,len(data))] = 1
                c = sum(where)
            counts.append(c)
            vectors = array([data[j] for j in find(where)])
            means[i] = average(vectors,axis=0)
            sigmas[i] = sqrt(var(vectors,axis=0))
            if sreg!=0: sigmas[i] = spatially_regularize(sigmas[i],sreg).ravel()
        if verbose>1:
            print "   sigmas",amin(sigmas),amax(sigmas),mean(sigmas),median(sigmas)
            print "   changes",sum(oldmins!=mins),reinit
            if iter>rstart: print "   outliers",outliers
        else:
            print sum(oldmins!=mins),
            sys.stdout.flush()
        # pylab.clf(); pylab.hist(sigmas.ravel()); pylab.draw(); pylab.ginput(1,timeout=0.1)
        sigmas = maximum(sigmas,reg)
        oldmins = mins
    if verbose>0: print
    return means,sigmas,counts

# GmmTree
# In order to avoid duplication of data vectors, GmmTree stores all data as a single
# big array and then just passes around shared array descriptors for the rows.
# Since the native code functions for nearest neighbor lookup don't want to deal with
# Python lists, they receive row indexes to identify which rows to compute distances
# from.  These row indexes are obtained from the descriptors via the self.row has
# table.

class GmmTree:
    def __init__(self,k=11,branch=None,ndata=None,mincount=None,reg=None,parent=None):
        self.k = k
        self.data = None # large floating point array containing the data vectors
        self.rows = None # rows of the data array, as separate array descriptors
        self.indexes = None # indexes of each row
        self.means = None # means for the children of this node
        self.sigmas = None # sigmas for the children of this node
        self.children = None # child nodes
        self.mean = None # mean for this node
        self.sigma = None # sigma for this node

        # used by IModel interface
        self.nfeatures = None # number of features after extraction
        self.collect = [] # feature vectors for training
        self.values = [] # corresponding classes

        # default parameters (used only by updateModel)
        self.mincluster = 1000
        self.maxcluster = 10000
        self.extractor = None # feature extractor
        if parent is not None:
            # avoid duplicating the object
            self.extractor = parent.extractor
        else:
            self.setExtractor("ScaledImageExtractor")

    def setExtractor(self,s):
        if type(s)==str:
            self.extractor = ocropy.make_IExtractor(s)
        else:
            self.extractor = s
            
    def info(self,*args):
        """Show information about this tree node."""
        path = list(args)
        if self.children is None:
            print path,len(self.rows)
        else:
            for i in range(len(self.children)):
                self.children[i].info(*(path+[i]))

    def get_path(self,path):
        if path==[]: return self
        return self.children[path[0]].get_path(path[1:])

    def show(self,*args):
        path = list(args)
        if path==[]:
            if self.children is None:
                showgrid(self.rows)
            else:
                showgrid3(self.means,self.means,self.sigmas)
        else:
            self.children[path[0]].show(*path[1:])

    def build(self,data,values,mincluster=1000,maxcluster=10000):
        """Build a tree from a set of data vectors (rows) and corresponding
        values (classes).  This actually just stores the row ids and then
        passes it off to build1."""
        # store the original data
        self.data = data
        # a hash table mapping row descriptors to row indexes
        self.row = {}
        rows = []
        for i in range(len(data)):
            v = data[i]
            self.row[id(v)] = i
            rows.append(v)
        # now do the real tree building
        self.build1(rows,values,mincluster,maxcluster)
        # update the mean and sigma for the root node specially,
        # since build1 only does it for children
        self.mean = mean(data,axis=0)
        self.sigma = var(data,axis=0)**0.5
        self.sigma = maximum(0.01*amax(self.sigma),self.sigma)
        self.row = None

    def build1(self,rows,values,mincluster=1000,maxcluster=10000):
        """Build a tree from a set of data vectors (rows) and corresponding
        values (classes).  The selected rows are given by the rows array.
        This method assumes that the self.row table mapping rows to row indexes
        has already been setup."""
        self.rows = rows
        self.indexes = array([self.row[id(v)] for v in rows],'i')
        self.values = values
        if len(rows)>mincluster:
            k = max(3,int(sqrt(len(rows))))
            # compute k new means and sigmas using EM algorithm
            means,sigmas,counts = gmm_em(rows,k,verbose=1)
            # assign the data vectors to their closest means
            which = array([nnlib.argmindist_sig(means,sigmas,row) for row in rows])
            # create a list of child nodes
            children = [None]*len(means)
            # now run through each of the children and create child nodes
            for i in range(len(means)):
                indexes = find(which==i)
                nrows = select(rows,indexes)
                nvalues = select(values,indexes)
                # create the child
                children[i] = GmmTree(parent=self)
                # store the row-to-index mapping
                children[i].data = self.data
                children[i].row = self.row
                # now build the tree recursively
                children[i].build1(nrows,nvalues,mincluster,maxcluster)
                # for convenience, also store the mean and sigma used to build this child
                children[i].mean = means[i]
                children[i].sigma = sigmas[i]
            self.means = means
            self.sigmas = sigmas
            self.children = children

    def find_terminal(self,key,mean=None,sigma=None):
        """Recursive search for the closest tree node; returns the node,
        plus the last mean and sigma that led to this node."""
        if self.children is not None:
            index = nnlib.argmindist_sig(self.means,self.sigmas,key)
            return self.children[index].find_terminal(key,self.means[index],self.sigmas[index])
        else:
            return self,self.mean,self.sigma
    def cdists(self,key,k=None):
        """Find the distances of the k-closest samples to the key."""
        if k is None: k = self.k
        node,center,sigma = self.find_terminal(key)
        # This is the same as dists_sig1(array(self.rows),sigma,key)
        dists = nnlib.some_dists_sig1(self.indexes,self.data,sigma,key)
        kbest = argsort(dists)[:k]
        dists = dists[kbest]
        values = select(self.values,kbest)
        return zip(dists,values)

    def preprocess(self,image):
        if self.extractor is None:
            return ocropy.as_numpy(image,flip=0)
        else:
            image = ocropy.as_narray(image,flip=0)
            out = ocropy.floatarray()
            self.extractor.extract(out,image)
            return ocropy.as_numpy(out,flip=0)
        
    # IModel methods
    def name(self):
        return "GmmTree"
    def cadd(self,image,c):
        image = ocropy.as_numpy(image,flip=1)
        preprocessed = self.preprocess(image).ravel()
        # r = int(sqrt(len(preprocessed)))
        # from pylab import imshow,show
        # imshow(ocropy.as_numpy(preprocessed.reshape(r,r))); show()
        if self.nfeatures is not None: assert self.nfeatures==len(preprocessed)
        else: self.nfeatures = len(preprocessed)
        self.collect.append(preprocessed)
        self.values.append(c)
    def updateModel(self):
        self.data = array(self.collect)
        self.collect = None
        self.build(self.data,self.values,mincluster=self.mincluster,maxcluster=self.maxcluster)
    def coutputs(self,image,k=None):
        """Compute the outputs for the given input image.  The result is
        a list of (cls,probability) tuples. (NB: that's probability, not cost.)"""
        image = ocropy.as_numpy(image,flip=1)
        image = self.preprocess(image).ravel()
        assert self.nfeatures==len(image)
        if k is None: k = self.k
        dists = self.cdists(image,k=k)
        classes = [p[1] for p in dists]
        outputs = classhist(classes)
        total = 1.0*sum([p[0] for p in outputs])
        return [(p[1],p[0]/total) for p in outputs]

    # backwards compatibility with numerical values; for Python, we just
    # turn these into strings (rather than the other way around)
    def add(self,image,c):
        self.cadd(image,chr(c))
    def outputs(self,image,k=None):
        result = self.coutputs(image,k=Noen)
        return [(ord(p[0]),p[1]) for p in result]
