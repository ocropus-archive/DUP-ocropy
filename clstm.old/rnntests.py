from pylab import *
from scipy.ndimage import filters

default_ninput = 2
default_n = 29

def generate_threshold(n=default_n,ninput=default_ninput,threshold=0.5,example=0):
    "No temporal dependencies, just threshold of the sum of the inputs."
    x = rand(n,ninput)
    y = 1.0*(sum(x,axis=1)>threshold*ninput).reshape(n,1)
    return x,y

def generate_mod(n=default_n,ninput=default_ninput,m=3,example=0):
    "Generate a regular beat every m steps. The input is random."
    x = rand(n,ninput)
    y = 1.0*(arange(n,dtype='i')%m==0).reshape(n,1)
    return x,y

def generate_dmod(n=default_n,ninput=default_ninput,m=3,example=0):
    """Generate a regular beat every m steps, the input is random
    except for the first dimension, which contains a downbeat
    at the very beginning."""
    x = rand(n,ninput)
    y = 1.0*(arange(n,dtype='i')%m==0).reshape(n,1)
    x[:,0] = 0
    x[0,0] = 1
    return x,y

def generate_imod(n=default_n,ninput=default_ninput,m=3,p=0.2,example=0):
    """Generate an output for every m input pulses."""
    if example:
        x = array(arange(n)%4==1,'i')
    else:
        x = array(rand(n)<p,'i')
    y = (add.accumulate(x)%m==1)*x*1.0
    x = array(vstack([x]*ninput).T,'f')
    y = y.reshape(len(y),1)
    return x,y

def generate_smod(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Generate an output for every m input pulses. The input
    is band limited, so it's a little easier than generate_imod."""
    x = rand(n)
    x = filters.gaussian_filter(x,r)
    x = (x>roll(x,-1))*(x>roll(x,1))
    y = (add.accumulate(x)%m==1)*x*1.0
    x = array(vstack([x]*ninput).T,'f')
    y = y.reshape(len(y),1)
    return x,y

def generate_anbn(ninput=default_ninput,n=default_n,k=default_n//3,example=0):
    """A simple detector for a^nb^n. Note that this does not
    train the network to distinguish this langugage from other languages."""
    inputs = zeros(n)
    outputs = zeros(n)
    if example:
        l = n//3
    else:
        l = 1+int((k-1)*rand())
    inputs[:l] = 1
    outputs[2*l] = 1
    outputs = outputs.reshape(len(outputs),1)
    return vstack([inputs]*ninput).T,outputs

def generate_timing(ninput=default_ninput,n=default_n,t=5,example=0):
    """A simple timing related task: output a spike if no spike occurred within
    t time steps before."""
    x = 0
    inputs = []
    while x<n:
        inputs.append(x)
        x += max(1,0.5*t*randn()+t)
    inputs = [-999990]+inputs
    outputs = []
    for i in range(1,len(inputs)):
        if inputs[i]-inputs[i-1]>t:
            outputs.append(inputs[i])
    inputs = inputs[1:]
    xs = zeros((n,ninput))
    xs[inputs,:] = 1.0
    ys = zeros((n,1))
    ys[outputs,:] = 1.0
    return xs,ys


def generate_revtiming(ninput=default_ninput,n=default_n,t=5,example=0):
    """A simple timing related task: output a spike if no spike occurs within
    t time steps after. This cannot be learned using a causal model (it requires
    a reverse model)."""
    x = 0
    inputs = []
    while x<n:
        inputs.append(x)
        x += max(1,0.5*t*randn()+t)
    inputs = inputs+[999999]
    outputs = []
    for i in range(len(inputs)-1):
        if inputs[i+1]-inputs[i]>t:
            outputs.append(inputs[i])
    inputs = inputs[:-1]
    xs = zeros((n,ninput))
    xs[inputs,:] = 1.0
    ys = zeros((n,1))
    ys[outputs,:] = 1.0
    return xs,ys

def generate_biditiming(ninput=default_ninput,n=default_n,t=5,example=0):
    x = 0
    inputs = []
    while x<n:
        inputs.append(x)
        x += max(1,0.5*t*randn()+t)
    inputs = [-999999]+inputs+[999999]
    outputs = []
    for i in range(1,len(inputs)-1):
        if inputs[i+1]-inputs[i]>=t and inputs[i]-inputs[i-1]>=t:
            outputs.append(inputs[i])
    inputs = inputs[1:-1]
    xs = zeros((n,ninput))
    xs[inputs,:] = 1.0
    ys = zeros((n,1))
    ys[outputs,:] = 1.0
    return xs,ys

def detect_12(x):
    n = len(x)
    y = zeros(n)
    state = 0
    for i in range(n):
        s = tuple(1*(x[i]>0.5))
        if s==(0,0): pass
        elif s==(1,0): state = 1
        elif s==(0,1) and state==1:
            y[i] = 1
            state = 0
        else: state = 0
    return y

def generate_detect(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Generates a random sequence of bits and outputs a "1" whenever there is
    a sequence of inputs 01-00*-10"""
    x = rand(n,2)
    x = filters.gaussian_filter(x,(r,0))
    x = 1.0*(x>roll(x,-1,0))*(x>roll(x,1,0))
    y = detect_12(x)
    return x,y.reshape(len(y),1)

def generate_revdetect(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Reverse of generate_detect."""
    xs,ys = generate_detect(n=n,ninput=ninput,m=m,r=r,example=example)
    return array(xs)[::-1],array(ys)[::-1]

def generate_bididetect(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Generate a particular pattern whenever there is some input trigger."""
    xs,ys = generate_detect(n=n,ninput=ninput,m=m,r=r,example=example)
    rys = detect_12(xs[::-1])[::-1].reshape(len(ys),1)
    return array(xs),array(ys*rys)

def generate_predict_and_sync():
    """Similar to smod, but the correct output is provided one step after
    the required prediction for resynchronization."""
    pass

def generate_distracted_recall():
    """Distracted sequence recall example."""
    pass

def generate_morse():
    """Morse code encoding/decoding."""
    pass

def genseq_timing1(n=30,threshold=0.2,m=4,example=0):
    """Returns an output for every input within m time steps.
    A 1 -> N -> 1 problem."""
    x = (rand(n)<threshold)
    l = find(x)
    y = zeros(len(x))
    for i in range(1,len(l)):
        if l[i]-l[i-1]<m: y[l[i]] = 1
    return (1.0*x).reshape(n,1),y.reshape(n,1)

def genseq_threshold1(n=30,d=3,threshold=0.5,c=0,scale=1.0):
    """Threshold on the first component only."""
    x = randn(n,d)
    y = (1.0*(x[:,c]>threshold)).reshape(len(x),1)
    x[:,c] *= scale
    return x,y

def genseq_delay(n=30,threshold=0.2,d=1):
    """Returns an output for every input within m time steps.
    A 1 -> N -> 1 problem."""
    x = array(rand(n)<threshold,'f')
    y = roll(x,d)
    if d>0: y[:d] = 0
    elif d<0: y[d:] = 0
    return x.reshape(n,1),y.reshape(n,1)

import h5py
import numpy as np

class H5SeqData:
    def __init__(self,fname,N=None):
        self.fname = fname
        h5 = h5py.File("rnntest-"+fname+".h5","w")
        self.h5 = h5
        dt = h5py.special_dtype(vlen=np.dtype('float32'))
        it = np.dtype('int32')
        self.inputs = h5.create_dataset("inputs",(1,), maxshape=(None,),compression="gzip",dtype=dt)
        self.inputs_dims = h5.create_dataset("inputs_dims",(1,2), maxshape=(None,2), dtype=it)
        self.outputs = h5.create_dataset("outputs",(1,),maxshape=(None,),compression="gzip",dtype=dt)
        self.outputs_dims = h5.create_dataset("outputs_dims",(1,2), maxshape=(None,2), dtype=it)
        self.fill = 0
        if N is not None: self.resize(N)
    def close(self):
        self.h5.close()
        self.h5 = None
    def __enter__(self):
        print "writing",self.fname
        return self
    def __exit__(self, type, value, traceback):
        self.close()
        print "done writing",self.fname
    def resize(self,n):
        self.inputs.resize((n,))
        self.inputs_dims.resize((n,2))
        self.outputs.resize((n,))
        self.outputs_dims.resize((n,2))
    def add(self,inputs,outputs):
        self.inputs[self.fill] = inputs.ravel()
        self.inputs_dims[self.fill] = array(inputs.shape,'i')
        self.outputs[self.fill] = outputs.ravel()
        self.outputs_dims[self.fill] = array(outputs.shape,'i')
        self.fill += 1

N = 50000

with H5SeqData("threshold",N) as db:
    for i in range(N):
        xs,ys = generate_threshold()
        db.add(xs,ys)

with H5SeqData("mod3",N) as db:
    for i in range(N):
        xs,ys = generate_mod(m=3)
        db.add(xs,ys)

with H5SeqData("dmod3",N) as db:
    for i in range(N):
        xs,ys = generate_dmod(m=3)
        db.add(xs,ys)

with H5SeqData("dmod4",N) as db:
    for i in range(N):
        xs,ys = generate_dmod(m=4)
        db.add(xs,ys)

with H5SeqData("dmod5",N) as db:
    for i in range(N):
        xs,ys = generate_dmod(m=5)
        db.add(xs,ys)

with H5SeqData("dmod6",N) as db:
    for i in range(N):
        xs,ys = generate_dmod(m=6)
        db.add(xs,ys)

with H5SeqData("imod3",N) as db:
    for i in range(N):
        xs,ys = generate_imod(m=3)
        db.add(xs,ys)

with H5SeqData("smod3",N) as db:
    for i in range(N):
        xs,ys = generate_smod(m=3)
        db.add(xs,ys)

with H5SeqData("smod4",N) as db:
    for i in range(N):
        xs,ys = generate_smod(m=4)
        db.add(xs,ys)

with H5SeqData("smod5",N) as db:
    for i in range(N):
        xs,ys = generate_smod(m=5)
        db.add(xs,ys)

with H5SeqData("anbn",N) as db:
    for i in range(N):
        xs,ys = generate_anbn()
        db.add(xs,ys)

with H5SeqData("timing",N) as db:
    for i in range(N):
        xs,ys = generate_timing()
        db.add(xs,ys)

with H5SeqData("revtiming",N) as db:
    for i in range(N):
        xs,ys = generate_revtiming()
        db.add(xs,ys)

with H5SeqData("biditiming",N) as db:
    for i in range(N):
        xs,ys = generate_biditiming()
        db.add(xs,ys)

with H5SeqData("detect",N) as db:
    for i in range(N):
        xs,ys = generate_detect()
        db.add(xs,ys)

with H5SeqData("revdetect",N) as db:
    for i in range(N):
        xs,ys = generate_revdetect()
        db.add(xs,ys)

with H5SeqData("bididetect",N) as db:
    for i in range(N):
        xs,ys = generate_bididetect()
        db.add(xs,ys)

with H5SeqData("delay1",N) as db:
    for i in range(N):
        xs,ys = genseq_delay(d=1)
        db.add(xs,ys)

with H5SeqData("delay2",N) as db:
    for i in range(N):
        xs,ys = genseq_delay(d=2)
        db.add(xs,ys)

with H5SeqData("delay3",N) as db:
    for i in range(N):
        xs,ys = genseq_delay(d=3)
        db.add(xs,ys)

with H5SeqData("rdelay1",N) as db:
    for i in range(N):
        xs,ys = genseq_delay(d=-1)
        db.add(xs,ys)

with H5SeqData("rdelay2",N) as db:
    for i in range(N):
        xs,ys = genseq_delay(d=-2)
        db.add(xs,ys)
