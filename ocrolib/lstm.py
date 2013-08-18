import common as ocrolib
import pdb
from pylab import *
import sys
from collections import defaultdict
from ocrolib.native import *
from ocrolib import edist
import nutils
import unicodedata

initial_range = 0.1

class RangeError(Exception):
    def __init__(self,s=None):
        Exception.__init__(self,s)

def prepare_line(line,pad=16):
    """Prepare a line for recognition; this inverts it, transposes
    it, and pads it."""
    line = line * 1.0/amax(line)
    line = amax(line)-line
    line = line.T
    if pad>0:
        w = line.shape[1]
        line = vstack([zeros((pad,w)),line,zeros((pad,w))])
    return line

def randu(*shape):
    # ATTENTION: whether you use randu or randn can make a difference.
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution."""
    return 2*rand(*shape)-1

def sigmoid(x):
    """Compute the sigmoid function.
    We don't bother with clipping the input value because IEEE floating
    point behaves reasonably with this function even for infinities."""
    return 1.0/(1.0+exp(-x))

def rownorm(a):
    """Compute a vector consisting of the Euclidean norm of the
    rows of the 2D array."""
    return sum(array(a)**2,axis=1)**.5

def check_nan(*args,**kw):
    "Check whether there are any NaNs in the argument arrays."
    for arg in args:
        if isnan(arg).any():
            raise FloatingPointError()

def sumouter(us,vs,lo=-1.0,hi=1.0,out=None):
    """Sum the outer products of the `us` and `vs`.
    Values are clipped into the range `[lo,hi]`.
    This is mainly used for computing weight updates
    in logistic regression layers."""
    result = out or zeros((len(us[0]),len(vs[0])))
    for u,v in zip(us,vs):
        result += outer(clip(u,lo,hi),v)
    return result

def sumprod(us,vs,lo=-1.0,hi=1.0,out=None):
    """Sum the element-wise products of the `us` and `vs`.
    Values are clipped into the range `[lo,hi]`.
    This is mainly used for computing weight updates
    in logistic regression layers."""
    assert len(us[0])==len(vs[0])
    result = out or zeros(len(us[0]))
    for u,v in zip(us,vs):
        result += clip(u,lo,hi)*v
    return result

class Network:
    """General interface for networks. This mainly adds convenience
    functions for `predict` and `train`."""
    def predict(self,xs):
        return self.forward(xs)
    def train(self,xs,ys,debug=0):
        xs = array(xs)
        ys = array(ys)
        pred = array(self.forward(xs))
        deltas = ys - pred
        self.backward(deltas)
        self.update()
        return pred
    def ctrain(self,xs,cs,debug=0,lo=1e-5,accelerated=1):
        """Training for classification.  This handles
        the special case of just two classes. It also
        can use regular least square error training or
        accelerated training using 1/pred as the error signal."""
        assert len(cs.shape)==1
        assert (cs==array(cs,'i')).all()
        xs = array(xs)
        pred = array(self.forward(xs))
        deltas = zeros(pred.shape)
        assert len(deltas)==len(cs)
        # NB: these deltas are such that they can be used
        # directly to update the gradient; some other libraries
        # use the negative value.
        if accelerated:
            # ATTENTION: These deltas use an "accelerated" error signal.
            if deltas.shape[1]==1:
                # Binary class case uses just one output variable.
                for i,c in enumerate(cs):
                    if c==0:
                        deltas[i,0] = -1.0/max(lo,1.0-pred[i,0])
                    else:
                        deltas[i,0] = 1.0/max(lo,pred[i,0])
            else:
                # For the multi-class case, we use all output variables.
                deltas[:,:] = -pred[:,:]
                for i,c in enumerate(cs):
                    deltas[i,c] = 1.0/max(lo,pred[i,c])
        else:
            # These are the deltas from least-square error
            # updates. They are slower than `accelerated`,
            # but may give more accurate probability estimates.
            if deltas.shape[1]==1:
                # Binary class case uses just one output variable.
                for i,c in enumerate(cs):
                    if c==0:
                        deltas[i,0] = -pred[i,0]
                    else:
                        deltas[i,0] = 1.0-pred[i,0]
            else:
                # For the multi-class case, we use all output variables.
                deltas[:,:] = -pred[:,:]
                for i,c in enumerate(cs):
                    deltas[i,c] = 1.0-pred[i,c]
        self.backward(deltas)
        self.update()
        return pred
    def setLearningRate(self,r,momentum=0.9):
        """Set the learning rate and momentum for weight updates."""
        self.learning_rate = r
        self.momentum = momentum
    def weights(self):
        """Return an iterator that iterates over (W,DW,name) triples
        representing the weight matrix, the computed deltas, and the names
        of all the components of this network. This needs to be implemented
        in subclasses. The objects returned by the iterator must not be copies,
        since they are updated in place by the `update` method."""
        pass
    def allweights(self):
        """Return all weights as a single vector. This is mainly a convenience
        function for plotting."""
        aw = list(self.weights())
        weights,derivs,names = zip(*aw)
        weights = [w.ravel() for w in weights]
        derivs = [d.ravel() for d in derivs]
        return concatenate(weights),concatenate(derivs)
    def update(self):
        """Update the weights using the deltas computed in the last forward/backward pass.
        Subclasses need not implement this, they should implement the `weights` method."""
        if not hasattr(self,"verbose"):
            self.verbose = 0
        if not hasattr(self,"deltas") or self.deltas is None:
            self.deltas = [zeros(dw.shape) for w,dw,n in self.weights()]
        for ds,(w,dw,n) in zip(self.deltas,self.weights()):
            ds.ravel()[:] = self.momentum * ds.ravel()[:] + self.learning_rate * dw.ravel()[:]
            w.ravel()[:] += ds.ravel()[:]
            if self.verbose:
                print n,(amin(w),amax(w)),(amin(dw),amax(dw))

class Logreg(Network):
    """A logistic regression network."""
    def __init__(self,Nh,No,initial_range=initial_range,rand=rand):
        self.Nh = Nh
        self.No = No
        self.W2 = randu(No,Nh+1)*initial_range
        self.DW2 = zeros((No,Nh+1))
    def ninputs(self):
        return self.Nh
    def noutputs(self):
        return self.No
    def forward(self,ys):
        n = len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = concatenate([ones(1),ys[i]])
            zs[i] = sigmoid(dot(self.W2,inputs[i]))
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * zs[i] * (1-zs[i])
            dys[i] = dot(dzspre[i],self.W2)[1:]
        self.dzspre = dzspre
        self.DW2 = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W2".split())
        for v in vars:
            a = array(getattr(self,v))
            print v,a.shape,amin(a),amax(a)
    def weights(self):
        yield self.W2,self.DW2,"Logreg"

class Softmax(Network):
    """A logistic regression network."""
    def __init__(self,Nh,No,initial_range=initial_range,rand=rand):
        self.Nh = Nh
        self.No = No
        self.W2 = randu(No,Nh+1)*initial_range
        self.DW2 = zeros((No,Nh+1))
    def ninputs(self):
        return self.Nh
    def noutputs(self):
        return self.No
    def forward(self,ys):
        n = len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = concatenate([ones(1),ys[i]])
            temp = dot(self.W2,inputs[i])
            temp = exp(clip(temp,-100,100))
            temp /= sum(temp)
            zs[i] = temp
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i]
            dys[i] = dot(dzspre[i],self.W2)[1:]
        self.DW2 = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W2".split())
        for v in vars:
            a = array(getattr(self,v))
            print v,a.shape,amin(a),amax(a)
    def weights(self):
        yield self.W2,self.DW2,"Softmax"

class MLP(Network):
    """A multilayer perceptron (direct implementation)."""
    def __init__(self,Ni,Nh,No,initial_range=initial_range,rand=randu):
        self.Ni = Ni
        self.Nh = Nh
        self.No = No
        self.W1 = rand(Nh,Ni+1)*initial_range
        self.W2 = rand(No,Nh+1)*initial_range
    def ninputs(self):
        return self.Ni
    def noutputs(self):
        return self.No
    def forward(self,xs):
        n = len(xs)
        inputs,ys,zs = [None]*n,[None]*n,[None]*n
        for i in range(n):
            inputs[i] = concatenate([ones(1),xs[i]])
            ys[i] = sigmoid(dot(self.W1,inputs[i]))
            ys[i] = concatenate([ones(1),ys[i]])
            zs[i] = sigmoid(dot(self.W2,ys[i]))
        self.state = (inputs,ys,zs)
        return zs
    def backward(self,deltas):
        xs,ys,zs = self.state
        n = len(xs)
        dxs,dyspre,dzspre,dys = [None]*n,[None]*n,[None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * zs[i] * (1-zs[i])
            dys[i] = dot(dzspre[i],self.W2)[1:]
            dyspre[i] = dys[i] * (ys[i] * (1-ys[i]))[1:]
            dxs[i] = dot(dyspre[i],self.W1)[1:]
        self.DW2 = sumouter(dzspre,ys)
        self.DW1 = sumouter(dyspre,xs)
        return dxs
    def weights(self):
        yield self.W1,self.DW1,"MLP1"
        yield self.W2,self.DW2,"MLP2"

# These are the nonlinearities used by the LSTM network.
# We don't bother parameterizing them here

def ffunc(x):
    "Nonlinearity used for gates."
    return 1.0/(1.0+exp(-x))
def fprime(x,y=None):
    "Derivative of nonlinearity used for gates."
    if y is None: y = sigmoid(x)
    return y*(1.0-y)
def gfunc(x):
    "Nonlinearity used for input to state."
    return tanh(x)
def gprime(x,y=None):
    "Derivative of nonlinearity used for input to state."
    if y is None: y = tanh(x)
    return 1-y**2
# ATTENTION: try linear for hfunc
def hfunc(x):
    "Nonlinearity used for output."
    return tanh(x)
def hprime(x,y=None):
    "Derivative of nonlinearity used for output."
    if y is None: y = tanh(x)
    return 1-y**2

# These two routines have been factored out of the class in order to 
# make their conversion to native code easy; these are the "inner loops"
# of the LSTM algorithm.

# Both functions are a straightforward implementation of the
# LSTM equations. It is possible to abstract this further and
# represent gates and memory cells as individual data structures.
# However, that is several times slower and the extra abstraction
# isn't actually all that useful.

def forward_py(n,N,ni,ns,na,xs,source,gix,gfx,gox,cix,gi,gf,go,ci,state,output,WGI,WGF,WGO,WCI,WIP,WFP,WOP):
    """Perform forward propagation of activations for a simple LSTM layer."""
    for t in range(n):
        prev = zeros(ns) if t==0 else output[t-1]
        source[t,0] = 1
        source[t,1:1+ni] = xs[t]
        source[t,1+ni:] = prev
        dot(WGI,source[t],out=gix[t])
        dot(WGF,source[t],out=gfx[t])
        dot(WGO,source[t],out=gox[t])
        dot(WCI,source[t],out=cix[t])
        if t>0:
            gix[t] += WIP*state[t-1]
            gfx[t] += WFP*state[t-1]
        gi[t] = ffunc(gix[t])
        gf[t] = ffunc(gfx[t])
        ci[t] = gfunc(cix[t])
        state[t] = ci[t]*gi[t]
        if t>0:
            state[t] += gf[t]*state[t-1]
            gox[t] += WOP*state[t]
        go[t] = ffunc(gox[t])
        output[t] = hfunc(state[t]) * go[t]
    assert not isnan(output[:n]).any()


def backward_py(n,N,ni,ns,na,deltas,
    """Perform backward propagation of deltas for a simple LSTM layer."""
                    source,
                    gix,gfx,gox,cix,
                    gi,gf,go,ci,
                    state,output,
                    WGI,WGF,WGO,WCI,
                    WIP,WFP,WOP,
                    sourceerr,
                    gierr,gferr,goerr,cierr,
                    stateerr,outerr,
                    DWGI,DWGF,DWGO,DWCI,
                    DWIP,DWFP,DWOP):
    for t in reversed(range(n)):
        outerr[t] = deltas[t]
        if t<n-1:
            outerr[t] += sourceerr[t+1][-ns:]
        goerr[t] = fprime(None,go[t]) * hfunc(state[t]) * outerr[t]
        stateerr[t] = hprime(state[t]) * go[t] * outerr[t]
        stateerr[t] += goerr[t]*WOP
        if t<n-1:
            stateerr[t] += gferr[t+1]*WFP
            stateerr[t] += gierr[t+1]*WIP
            stateerr[t] += stateerr[t+1]*gf[t+1]
        if t>0:
            gferr[t] = fprime(None,gf[t])*stateerr[t]*state[t-1]
        gierr[t] = fprime(None,gi[t])*stateerr[t]*ci[t] # gfunc(cix[t])
        cierr[t] = gprime(None,ci[t])*stateerr[t]*gi[t]
        dot(gierr[t],WGI,out=sourceerr[t])
        if t>0:
            sourceerr[t] += dot(gferr[t],WGF)
        sourceerr[t] += dot(goerr[t],WGO)
        sourceerr[t] += dot(cierr[t],WCI)
    DWIP = nutils.sumprod(gierr[1:n],state[:n-1],out=DWIP)
    DWFP = nutils.sumprod(gferr[1:n],state[:n-1],out=DWFP)
    DWOP = nutils.sumprod(goerr[:n],state[:n],out=DWOP)
    DWGI = nutils.sumouter(gierr[:n],source[:n],out=DWGI)
    DWGF = nutils.sumouter(gferr[1:n],source[1:n],out=DWGF)
    DWGO = nutils.sumouter(goerr[:n],source[:n],out=DWGO)
    DWCI = nutils.sumouter(cierr[:n],source[:n],out=DWCI)

class LSTM(Network):
    """A standard LSTM network. This is a direct implementation of all the forward
    and backward propagation formulas, mainly for speed. (There is another, more
    abstract implementation as well, but that's significantly slower in Python
    due to function call overhead.)"""
    def __init__(self,ni,ns,initial=initial_range,maxlen=5000):
        na = 1+ni+ns
        self.dims = ni,ns,na
        self.init_weights(initial)
        self.allocate(maxlen)
    def ninputs(self):
        return self.dims[0]
    def noutputs(self):
        return self.dims[1]
    def states(self):
        """Return the internal state array for the last forward
        propagation. This is mostly used for visualizations."""
        return array(self.state[:self.last_n])
    def init_weights(self,initial):
        "Initialize the weight matrices and derivatives"
        ni,ns,na = self.dims
        # gate weights
        for w in "WGI WGF WGO WCI".split():
            setattr(self,w,randu(ns,na)*initial)
            setattr(self,"D"+w,zeros((ns,na)))
        # peep weights
        for w in "WIP WFP WOP".split():
            setattr(self,w,randu(ns)*initial)
            setattr(self,"D"+w,zeros(ns))
    def weights(self):
        "Yields all the weight and derivative matrices"
        weights = "WGI WGF WGO WCI WIP WFP WOP"
        for w in weights.split():
            yield(getattr(self,w),getattr(self,"D"+w),w)
    def info(self):
        "Print info about the internal state"
        vars = "WGI WGF WGO WIP WFP WOP cix ci gix gi gox go gfx gf"
        vars += " source state output gierr gferr goerr cierr stateerr"
        vars = vars.split()
        vars = sorted(vars)
        for v in vars:
            a = array(getattr(self,v))
            print v,a.shape,amin(a),amax(a)
    def allocate(self,n):
        """Allocate space for the internal state variables.
        `n` is the maximum sequence length that can be processed."""
        ni,ns,na = self.dims
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        for v in vars.split():
            setattr(self,v,nan*ones((n,ns)))
        self.source = nan*ones((n,na))
        self.sourceerr = nan*ones((n,na))
    def reset(self,n):
        """Reset the contents of the internal state variables to `nan`"""
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        vars += " source sourceerr"
        for v in vars.split():
            getattr(self,v)[:,:] = nan
    def forward(self,xs):
        """Perform forward propagation of activations."""
        ni,ns,na = self.dims
        assert len(xs[0])==ni
        n = len(xs)
        self.last_n = n
        N = len(self.gi)
        if n>N: raise ocrolib.RecognitionError("input too large for LSTM model")
        self.reset(n)
        forward_py(n,N,ni,ns,na,xs,
                   self.source,
                   self.gix,self.gfx,self.gox,self.cix,
                   self.gi,self.gf,self.go,self.ci,
                   self.state,self.output,
                   self.WGI,self.WGF,self.WGO,self.WCI,
                   self.WIP,self.WFP,self.WOP)
        assert not isnan(self.output[:n]).any()
        return self.output[:n]
    def backward(self,deltas):
        """Perform backward propagation of deltas."""
        ni,ns,na = self.dims
        n = len(deltas)
        self.last_n = n
        N = len(self.gi)
        if n>N: raise ocrolib.RecognitionError("input too large for LSTM model")
        backward_py(n,N,ni,ns,na,deltas,
                    self.source,
                    self.gix,self.gfx,self.gox,self.cix,
                    self.gi,self.gf,self.go,self.ci,
                    self.state,self.output,
                    self.WGI,self.WGF,self.WGO,self.WCI,
                    self.WIP,self.WFP,self.WOP,
                    self.sourceerr,
                    self.gierr,self.gferr,self.goerr,self.cierr,
                    self.stateerr,self.outerr,
                    self.DWGI,self.DWGF,self.DWGO,self.DWCI,
                    self.DWIP,self.DWFP,self.DWOP)
        return [s[1:1+ni] for s in self.sourceerr[:n]]

################################################################
# combination classifiers
################################################################


class Stacked(Network):
    """Stack two networks on top of each other."""
    def __init__(self,nets):
        self.nets = nets
        self.dstats = defaultdict(list)
    def ninputs(self):
        return self.nets[0].ninputs()
    def noutputs(self):
        return self.nets[-1].noutputs()
    def forward(self,xs):
        for i,net in enumerate(self.nets):
            xs = net.forward(xs)
        return xs
    def backward(self,deltas):
        self.ldeltas = [deltas]
        for i,net in reversed(list(enumerate(self.nets))):
            if deltas is not None:
                self.dstats[i].append((amin(deltas),mean(deltas),amax(deltas)))
            deltas = net.backward(deltas)
            self.ldeltas.append(deltas)
        self.ldeltas = self.ldeltas[::-1]
        return deltas
    def lastdeltas(self):
        return self.ldeltas[-1]
    def info(self):
        for net in self.nets:
            net.info()
    def states(self):
        return self.nets[0].states()
    def weights(self):
        for i,net in enumerate(self.nets):
            for w,dw,n in net.weights():
                yield w,dw,"Stacked%d/%s"%(i,n)

class Reversed(Network):
    """Run a network on the time-reversed input."""
    def __init__(self,net):
        self.net = net
    def ninputs(self):
        return self.net.ninputs()
    def noutputs(self):
        return self.net.noutputs()
    def forward(self,xs):
        return self.net.forward(xs[::-1])[::-1]
    def backward(self,deltas):
        result = self.net.backward(deltas[::-1])
        return result[::-1] if result is not None else None
    def info(self):
        self.net.info()
    def states(self):
        return self.net.states()[::-1]
    def weights(self):
        for w,dw,n in self.net.weights():
            yield w,dw,"Reversed/%s"%n

class Parallel(Network):
    """Run multiple networks in parallel on the same input."""
    def __init__(self,*nets):
        self.nets = nets
    def forward(self,xs):
        outputs = [net.forward(xs) for net in self.nets]
        outputs = zip(*outputs)
        outputs = [concatenate(l) for l in outputs]
        return outputs
    def backward(self,deltas):
        deltas = array(deltas)
        start = 0
        for i,net in enumerate(self.nets):
            k = net.noutputs()
            net.backward(deltas[:,start:start+k])
            start += k
        return None
    def info(self):
        for net in self.nets:
            net.info()
    def states(self):
        states = [net.states() for net in self.nets]
        outputs = zip(*outputs)
        outputs = [concatenate(l) for l in outputs]
        return outputs
    def weights(self):
        for i,net in enumerate(self.nets):
            for w,dw,n in net.weights():
                yield w,dw,"Parallel%d/%s"%(i,n)

def MLP1(Ni,Ns,No):
    """An MLP implementation by stacking two `Logreg` networks on top
    of each other."""
    lr1 = Logreg(Ni,Ns)
    lr2 = Logreg(Ns,No)
    stacked = Stacked([lr1,lr2])
    return stacked

def LSTM1(Ni,Ns,No):
    """An LSTM layer with a `Logreg` layer for the output."""
    lstm = LSTM(Ni,Ns)
    if No==1:
        logreg = Logreg(Ns,No)
    else:
        logreg = Softmax(Ns,No)
    stacked = Stacked([lstm,logreg])
    return stacked

def BIDILSTM(Ni,Ns,No):
    """A bidirectional LSTM, constructed from regular and reversed LSTMs."""
    lstm1 = LSTM(Ni,Ns)
    lstm2 = Reversed(LSTM(Ni,Ns))
    bidi = Parallel(lstm1,lstm2)
    assert No>1
    # logreg = Logreg(2*Ns,No)
    logreg = Softmax(2*Ns,No)
    stacked = Stacked([bidi,logreg])
    return stacked

################################################################
# LSTM classification with forward/backward alignment ("CTC")
################################################################

def make_target(cs,nc):
    """Given a list of target classes `cs` and a total
    maximum number of classes, compute an array that has
    a `1` in each column and time step corresponding to the
    target class."""
    result = zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0] = 1.0
        result[2*i+1,j] = 1.0
    result[-1,0] = 1.0
    return result

def translate_back0(outputs,threshold=0.25):
    """Simple code for translating output from a classifier
    back into a list of classes. TODO/ATTENTION: this can
    probably be improved."""
    ms = amax(outputs,axis=1)
    cs = argmax(outputs,axis=1)
    cs[ms<threshold*amax(outputs)] = 0
    result = []
    for i in range(1,len(cs)):
        if cs[i]!=cs[i-1]:
            if cs[i]!=0:
                result.append(cs[i])
    return result

from scipy.ndimage import measurements,filters

def translate_back(outputs,threshold=0.7,pos=0):
    """Translate back. Thresholds on class 0, then assigns
    the maximum class to each region."""
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,arange(1,amax(mask)+1))
    if pos: return maxima
    return [c for (r,c) in maxima]

def log_mul(x,y):
    "Perform multiplication in the log domain (i.e., addition)."
    return x+y

def log_add(x,y):
    "Perform addition in the log domain."
    #return where(abs(x-y)>10,maximum(x,y),log(exp(x-y)+1)+y)
    return where(abs(x-y)>10,maximum(x,y),log(exp(clip(x-y,-20,20))+1)+y)

def forward_algorithm(match,skip=-5.0):
    """Apply the forward algorithm to an array of log state
    correspondence probabilities."""
    v = skip*arange(len(match[0]))
    result = []
    for i in range(0,len(match)):
        w = roll(v,1).copy()
        w[0] = skip*i
        v = log_add(log_mul(v,match[i]),log_mul(w,match[i]))
        result.append(v)
    return array(result,'f')

def forwardbackward(lmatch):
    """Apply the forward-backward algorithm to an array of log state
    correspondence probabilities."""
    lr = forward_algorithm(lmatch)
    rl = forward_algorithm(lmatch[::-1,::-1])[::-1,::-1]
    both = lr+rl
    return both

def ctc_align_targets(outputs,targets,threshold=100.0,verbose=0,debug=0,lo=1e-5):
    """Perform alignment between the `outputs` of a neural network
    classifier and a list of `targets`."""
    outputs = maximum(lo,outputs)
    outputs = outputs * 1.0/sum(outputs,axis=1)[:,newaxis]
    match = dot(outputs,targets.T)
    lmatch = log(match)
    if debug:
        figure("ctcalign"); clf();
        subplot(411); imshow(outputs.T,interpolation='nearest',cmap=cm.hot)
        subplot(412); imshow(lmatch.T,interpolation='nearest',cmap=cm.hot)
    assert not isnan(lmatch).any()
    both = forwardbackward(lmatch)
    epath = exp(both-amax(both))
    l = sum(epath,axis=0)[newaxis,:]
    epath /= where(l==0.0,1e-9,l)
    aligned = maximum(lo,dot(epath,targets))
    l = sum(aligned,axis=1)[:,newaxis]
    aligned /= where(l==0.0,1e-9,l)
    if debug:
        subplot(413); imshow(epath.T,cmap=cm.hot,interpolation='nearest')
        subplot(414); imshow(aligned.T,cmap=cm.hot,interpolation='nearest')
        ginput(1,0.01);
    return aligned

def normalize_nfkc(s):
    return unicodedata.normalize('NFKC',s)

def add_training_info(network):
    return network

class SeqRecognizer:
    """Perform sequence recognition using BIDILSTM and alignment."""
    def __init__(self,ninput,nstates,noutput=-1,codec=None,normalize=normalize_nfkc):
        self.Ni = ninput
        if codec: noutput = codec.size()
        assert noutput>0
        self.No = noutput
        self.lstm = BIDILSTM(ninput,nstates,noutput)
        self.setLearningRate(1e-4)
        self.debug_align = 0
        self.normalize = normalize
        self.codec = codec
        self.clear_log()
    def clear_log(self):
        self.command_log = []
        self.error_log = []
        self.cerror_log = []
        self.key_log = []
    def __setstate__(self,state):
        self.__dict__.update(state)
        self.upgrade()
    def upgrade(self):
        if "last_trial" not in dir(self): self.last_trial = 0
        if "command_log" not in dir(self): self.command_log = []
        if "error_log" not in dir(self): self.error_log = []
        if "cerror_log" not in dir(self): self.cerror_log = []
        if "key_log" not in dir(self): self.key_log = []
    def info(self):
        self.net.info()
    def setLearningRate(self,r,momentum=0.9):
        self.lstm.setLearningRate(r,momentum)
    def predictSequence(self,xs):
        "Predict an integer sequence of codes."
        assert xs.shape[1]==self.Ni,\
            "wrong image height (image: %d, expected: %d)"%(xs.shape[1],self.Ni)
        self.outputs = array(self.lstm.forward(xs))
        return translate_back(self.outputs)
    def trainSequence(self,xs,cs,update=1,key=None):
        "Train with an integer sequence of codes."
        assert xs.shape[1]==self.Ni,"wrong image height"
        # forward step
        self.outputs = array(self.lstm.forward(xs))
        # CTC alignment
        self.targets = array(make_target(cs,self.No))
        self.aligned = array(ctc_align_targets(self.outputs,self.targets,debug=self.debug_align))
        # propagate the deltas back
        deltas = self.aligned-self.outputs
        self.lstm.backward(deltas)
        if update: self.lstm.update()
        # translate back into a sequence
        result = translate_back(self.outputs)
        # compute least square error
        self.error = sum(deltas**2)
        self.error_log.append(self.error**.5/len(cs))
        # compute class error
        self.cerror = edist.levenshtein(cs,result)
        self.cerror_log.append((self.cerror,len(cs)))
        # training keys
        self.key_log.append(key)
        return result
    # we keep track of errors within the object; this even gets
    # saved to give us some idea of the training history
    def errors(self,range=10000,smooth=0):
        result = self.error_log[-range:]
        if smooth>0: result = filters.gaussian_filter(result,smooth,mode='mirror')
        return result
    def cerrors(self,range=10000,smooth=0):
        result = [e*1.0/max(1,n) for e,n in self.cerror_log[-range:]]
        if smooth>0: result = filters.gaussian_filter(result,smooth,mode='mirror')
        return result

    def s2l(self,s):
        "Convert a unicode sequence into a code sequence for training."
        s = self.normalize(s)
        s = [c for c in s]
        return self.codec.encode(s)
    def l2s(self,l):
        "Convert a code sequence into a unicode string after recognition."
        l = self.codec.decode(l)
        return u"".join(l)
    def trainString(self,xs,s,update=1):
        "Perform training with a string. This uses the codec and normalizer."
        return self.trainSequence(xs,self.s2l(s),update=update)
    def predictString(self,xs):
        "Predict output as a string. This uses codec and normalizer."
        cs = self.predictSequence(xs)
        return self.l2s(cs)

class Codec:
    """Translate between integer codes and characters."""
    def init(self,charset):
        charset = sorted(list(set(charset)))
        self.code2char = {}
        self.char2code = {}
        for code,char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        return self
    def size(self):
        """The total number of codes (use this for the number of output
        classes when training a classifier."""
        return len(list(self.code2char.keys()))
    def encode(self,s):
        "Encode the string `s` into a code sequence."
        tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in s]
    def decode(self,l):
        "Decode a code sequence into a string."
        s = [self.code2char.get(c,"~") for c in l]
        return s

ascii_labels = [""," ","~"] + [unichr(x) for x in range(33,126)]

def ascii_codec():
    "Create a codec containing just ASCII characters."
    return Codec().init(ascii_labels)

def ocropus_codec():
    """Create a codec containing ASCII characters plus the default
    character set from ocrolib."""
    import ocrolib
    base = [c for c in ascii_labels]
    base_set = set(base)
    extra = [c for c in ocrolib.chars.default if c not in base_set]
    return Codec().init(base+extra)
