# TODO:
# - handle UTF-8 inputs correctly

from pylab import *
from collections import Counter,defaultdict
import glob,re,heapq,os,cPickle

def method(cls):
    """Adds the function as a method to the given class."""
    import new
    def _wrap(f):
        cls.__dict__[f.func_name] = new.instancemethod(f,None,cls)
        return None
    return _wrap

replacements = [
    (r'[\0-\x1f]',''), # get rid of weird control characters
    (r'\s+',' '), # replace multiple spaces
    (r'[0-9]','9'), # don't try to model individual digit frequencies
    (r'[^-=A-Za-z0-9.,?:()"/\' ]','!'), # model other special characters just as '!'
    (r'``',"''"), # replace fancy double quotes
    (r'"',"''"), # replace fancy double quotes
    (r'[~]',""), # replace rejects with nothing
    ]

def lineproc(s):
    for regex,subst in replacements:
        s = re.sub(regex,subst,s)
    return s

def compute_ngraphs(fnames,n,lineproc=lineproc):
    counter = Counter()
    lineskip = 0
    linelimit = 2000
    for fnum,fname in enumerate(fnames):
        print fnum,"of",len(fnames),":",fname
        if fname.startswith("lineskip="):
            lineskip = int(fname.split("=")[1])
            print "changing lineskip to",lineskip
            continue
        if fname.startswith("linelimit="):
            linelimit = int(fname.split("=")[1])
            print "changing linelimit to",linelimit
            continue
        with open(fname) as stream:
            for lineno,line in enumerate(stream.xreadlines()):
                if lineno<lineskip: continue
                if lineno>=linelimit+lineskip: break
                line = line[:-1]
                if len(line)<3: continue
                line = lineproc(line)
                line = "_"*(n-1)+line+"_"*(n-1)
                for i in range(len(line)-n):
                    sub = line[i:i+n]
                    counter[sub] += 1
    return counter

class NGraphs:
    def __init__(self,N=4):
        self.N = N

@method(NGraphs)
def computePosteriors(self,counter):
    self.N = len(counter.items()[0][0])
    ngrams = defaultdict(list)
    for k,v in counter.items():
        ngrams[k[:-1]].append((k,v))
    posteriors = {}
    for prefix in ngrams.keys():
        ps = [(k[-1],v) for k,v in ngrams[prefix]] + [("~",1)]
        total = sum([v for k,v in ps])
        total = log(total)
        ps = {k : total-log(v) for k,v in ps}
        posteriors[prefix] = ps
    self.posteriors = posteriors

def rsample(dist):
    v = add.accumulate(dist)
    assert abs(v[-1]-1)<1e-3
    val = rand()
    return searchsorted(v,val)

@method(NGraphs)
def buildFromFiles(self,fnames,n):
    print "reading",len(fnames),"files"
    counter = compute_ngraphs(fnames,n)
    print "got",sum(counter.values()),"%d-graphs"%(n,)
    self.computePosteriors(counter)
    print "done building posteriors"

@method(NGraphs)
def sample(self,n=80,prefix=None):
    if prefix is None:
        prefix = "_"*self.N
    for i in range(n):
        posteriors = self.posteriors.get(prefix[-self.N+1:])
        if posteriors is None:
            prefix += chr(ord("a")+int(rand()*26))
        else:
            items = [(k,p) for k,p in posteriors.items() if k not in ["~","_"]]
            items += [(" ",10.0)]
            ks = [k for k,p in items]
            ps = array([p for k,p in items],'f')
            ps = exp(-ps)
            ps /= sum(ps)
            j = rsample(ps)
            prefix += ks[j]
    return prefix[self.N:]

