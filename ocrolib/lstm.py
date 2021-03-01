# An implementation of LSTM networks, CTC alignment, and related classes.
#
# This code operates on sequences of vectors as inputs, and either outputs
# sequences of vectors, or symbol sequences. Sequences of vectors are
# represented as 2D arrays, with rows representing vectors at different
# time steps.
#
# The code makes liberal use of array programming, including slicing,
# both for speed and for simplicity. All arrays are actual narrays (not matrices),
# so `*` means element-wise multiplication. If you're not familiar with array
# programming style, the numerical code may be hard to follow. If you're familiar with
# MATLAB, here is a side-by-side comparison: http://wiki.scipy.org/NumPy_for_Matlab_Users
#
# Implementations follow the mathematical formulas for forward and backward
# propagation closely; these are not documented in the code, but you can find
# them in the original publications or the slides for the LSTM tutorial
# at http://lstm.iupr.com/
#
# You can find a simple example of how to use this code in this worksheet:
# https://docs.google.com/a/iupr.com/file/d/0B2VUW2Zx_hNoXzJQemFhOXlLN0U
# More complex usage is illustrated by the ocropus-rpred and ocropus-rtrain
# command line programs.
#
# Author: Thomas M. Breuel
# License: Apache 2.0

from __future__ import print_function

from collections import defaultdict
import unicodedata

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,filters

import common as ocrolib
from ocrolib.exceptions import RecognitionError
from ocrolib.edist import levenshtein
import utils

initial_range = 0.1

class RangeError(Exception):
    def __init__(self,s=None):
        Exception.__init__(self,s)

def prepare_line(line,pad=16):
    """Prepare a line for recognition; this inverts it, transposes
    it, and pads it."""
    line = line * 1.0/np.amax(line)
    line = np.amax(line)-line
    line = line.T
    if pad>0:
        w = line.shape[1]
        line = np.vstack([np.zeros((pad,w)),line,np.zeros((pad,w))])
    return line

def randu(*shape):
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution for weight initializations.
    Empirically, the choice of randu/randn can make a difference
    for neural network initialization."""
    return 2*np.random.rand(*shape)-1

def sigmoid(x):
    """Compute the sigmoid function.
    We don't bother with clipping the input value because IEEE floating
    point behaves reasonably with this function even for infinities."""
    return 1.0/(1.0+np.exp(-x))

def rownorm(a):
    """Compute a vector consisting of the Euclidean norm of the
    rows of the 2D array."""
    return np.sum(np.array(a)**2,axis=1)**.5

def check_nan(*args,**kw):
    "Check whether there are any NaNs in the argument arrays."
    for arg in args:
        if np.isnan(arg).any():
            raise FloatingPointError()

def sumouter(us,vs,lo=-1.0,hi=1.0,out=None):
    """Sum the outer products of the `us` and `vs`.
    Values are clipped into the range `[lo,hi]`.
    This is mainly used for computing weight updates
    in logistic regression layers."""
    result = out or np.zeros((len(us[0]),len(vs[0])))
    for u,v in zip(us,vs):
        result += np.outer(np.clip(u,lo,hi),v)
    return result

def graphemelist(str):
    l = list()
    s = ""
    for char in str:
        if unicodedata.category(char)[0] == 'M':
            s += char
        else:
            l.append(s)
            s = char
    l.append(s)
    return l

class Network:
    """General interface for networks. This mainly adds convenience
    functions for `predict` and `train`.

    For the purposes of this library, all inputs and outputs are
    in the form of (temporal) sequences of vectors. Sequences of
    vectors are represented as 2D arrays, with each row representing
    a vector at the time step given by the row index. Both activations
    and deltas are propagated that way.

    Common implementations of this are the `MLP`, `Logreg`, `Softmax`,
    and `LSTM` networks. These implementations do most of the numerical
    computation and learning.

    Networks are designed such that they can be abstracted; that is,
    you can create a network class that implements forward/backward
    methods but internally implements through calls to component networks.
    The `Stacked`, `Reversed`, and `Parallel` classes below take advantage
    of that.
    """

    def predict(self,xs):
        """Prediction is the same as forward propagation."""
        return self.forward(xs)

    def train(self,xs,ys,debug=0):
        """Training performs forward propagation, computes the output deltas
        as the difference between the predicted and desired values,
        and then propagates those deltas backwards."""
        xs = np.array(xs)
        ys = np.array(ys)
        pred = np.array(self.forward(xs))
        deltas = ys - pred
        self.backward(deltas)
        self.update()
        return pred

    def walk(self):
        yield self

    def preSave(self):
        pass

    def postLoad(self):
        pass

    def ctrain(self,xs,cs,debug=0,lo=1e-5,accelerated=1):
        """Training for classification.  This handles
        the special case of just two classes. It also
        can use regular least square error training or
        accelerated training using 1/pred as the error signal."""
        assert len(cs.shape)==1
        assert (cs==np.array(cs,'i')).all()
        xs = np.array(xs)
        pred = np.array(self.forward(xs))
        deltas = np.zeros(pred.shape)
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
        return np.concatenate(weights),np.concatenate(derivs)

    def update(self):
        """Update the weights using the deltas computed in the last forward/backward pass.
        Subclasses need not implement this, they should implement the `weights` method."""
        if not hasattr(self,"verbose"):
            self.verbose = 0
        if not hasattr(self,"deltas") or self.deltas is None:
            self.deltas = [np.zeros(dw.shape) for w,dw,n in self.weights()]
        for ds,(w,dw,n) in zip(self.deltas,self.weights()):
            ds.ravel()[:] = self.momentum * ds.ravel()[:] + self.learning_rate * dw.ravel()[:]
            w.ravel()[:] += ds.ravel()[:]
            if self.verbose:
                print(n, (np.amin(w), np.amax(w)), (np.amin(dw), np.amax(dw)))

''' The following are subclass responsibility:

    def forward(self,xs):
        """Propagate activations forward through the network.
        This needs to be implemented in subclasses.
        It updates the internal state of the object for an (optional)
        subsequent call to `backward`.
        """
        pass

    def backward(self,deltas):
        """Propagate error signals backward through the network.
        This needs to be implemented in subclasses.
        It assumes that activations for the input have previously
        been computed by a call to `forward`.
        It should not perform weight updates (that is handled by
        the `update` method)."""
        pass

'''

class Logreg(Network):
    """A logistic regression layer, a straightforward implementation
    of the logistic regression equations. Uses 1-augmented vectors."""
    def __init__(self,Nh,No,initial_range=initial_range,rand=np.random.rand):
        self.Nh = Nh
        self.No = No
        self.W2 = randu(No,Nh+1)*initial_range
        self.DW2 = np.zeros((No,Nh+1))
    def ninputs(self):
        return self.Nh
    def noutputs(self):
        return self.No
    def forward(self,ys):
        n = len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = np.concatenate([np.ones(1),ys[i]])
            zs[i] = sigmoid(np.dot(self.W2,inputs[i]))
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * zs[i] * (1-zs[i])
            dys[i] = np.dot(dzspre[i],self.W2)[1:]
        self.dzspre = dzspre
        self.DW2 = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W2".split())
        for v in vars:
            a = np.array(getattr(self,v))
            print(v, a.shape, np.amin(a), np.amax(a))
    def weights(self):
        yield self.W2,self.DW2,"Logreg"

class Softmax(Network):
    """A softmax layer, a straightforward implementation
    of the softmax equations. Uses 1-augmented vectors."""
    def __init__(self,Nh,No,initial_range=initial_range,rand=np.random.rand):
        self.Nh = Nh
        self.No = No
        self.W2 = randu(No,Nh+1)*initial_range
        self.DW2 = np.zeros((No,Nh+1))
    def ninputs(self):
        return self.Nh
    def noutputs(self):
        return self.No
    def forward(self,ys):
        """Forward propagate activations. This updates the internal
        state for a subsequent call to `backward` and returns the output
        activations."""
        n = len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = np.concatenate([np.ones(1),ys[i]])
            temp = np.dot(self.W2,inputs[i])
            temp = np.exp(np.clip(temp,-100,100))
            temp /= np.sum(temp)
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
            dys[i] = np.dot(dzspre[i],self.W2)[1:]
        self.DW2 = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W2".split())
        for v in vars:
            a = np.array(getattr(self,v))
            print(v, a.shape, np.amin(a), np.amax(a))
    def weights(self):
        yield self.W2,self.DW2,"Softmax"

class MLP(Network):
    """A multilayer perceptron (direct implementation). Effectively,
    two `Logreg` layers stacked on top of each other, or a simple direct
    implementation of the MLP equations. This is mainly used for testing."""
    def __init__(self,Ni,Nh,No,initial_range=initial_range,rand=randu):
        self.Ni = Ni
        self.Nh = Nh
        self.No = No
        self.W1 = np.random.rand(Nh,Ni+1)*initial_range
        self.W2 = np.random.rand(No,Nh+1)*initial_range
    def ninputs(self):
        return self.Ni
    def noutputs(self):
        return self.No
    def forward(self,xs):
        n = len(xs)
        inputs,ys,zs = [None]*n,[None]*n,[None]*n
        for i in range(n):
            inputs[i] = np.concatenate([np.ones(1),xs[i]])
            ys[i] = sigmoid(np.dot(self.W1,inputs[i]))
            ys[i] = np.concatenate([np.ones(1),ys[i]])
            zs[i] = sigmoid(np.dot(self.W2,ys[i]))
        self.state = (inputs,ys,zs)
        return zs
    def backward(self,deltas):
        xs,ys,zs = self.state
        n = len(xs)
        dxs,dyspre,dzspre,dys = [None]*n,[None]*n,[None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * zs[i] * (1-zs[i])
            dys[i] = np.dot(dzspre[i],self.W2)[1:]
            dyspre[i] = dys[i] * (ys[i] * (1-ys[i]))[1:]
            dxs[i] = np.dot(dyspre[i],self.W1)[1:]
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
    # cliping to avoid overflows
    return 1.0/(1.0+np.exp(np.clip(-x,-20,20)))
def fprime(x,y=None):
    "Derivative of nonlinearity used for gates."
    if y is None: y = sigmoid(x)
    return y*(1.0-y)
def gfunc(x):
    "Nonlinearity used for input to state."
    return np.tanh(x)
def gprime(x,y=None):
    "Derivative of nonlinearity used for input to state."
    if y is None: y = np.tanh(x)
    return 1-y**2
# ATTENTION: try linear for hfunc
def hfunc(x):
    "Nonlinearity used for output."
    return np.tanh(x)
def hprime(x,y=None):
    "Derivative of nonlinearity used for output."
    if y is None: y = np.tanh(x)
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
        prev = np.zeros(ns) if t==0 else output[t-1]
        source[t,0] = 1
        source[t,1:1+ni] = xs[t]
        source[t,1+ni:] = prev
        np.dot(WGI,source[t],out=gix[t])
        np.dot(WGF,source[t],out=gfx[t])
        np.dot(WGO,source[t],out=gox[t])
        np.dot(WCI,source[t],out=cix[t])
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
    assert not np.isnan(output[:n]).any()


def backward_py(n,N,ni,ns,na,deltas,
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
    """Perform backward propagation of deltas for a simple LSTM layer."""
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
        np.dot(gierr[t],WGI,out=sourceerr[t])
        if t>0:
            sourceerr[t] += np.dot(gferr[t],WGF)
        sourceerr[t] += np.dot(goerr[t],WGO)
        sourceerr[t] += np.dot(cierr[t],WCI)
    DWIP = utils.sumprod(gierr[1:n],state[:n-1],out=DWIP)
    DWFP = utils.sumprod(gferr[1:n],state[:n-1],out=DWFP)
    DWOP = utils.sumprod(goerr[:n],state[:n],out=DWOP)
    DWGI = utils.sumouter(gierr[:n],source[:n],out=DWGI)
    DWGF = utils.sumouter(gferr[1:n],source[1:n],out=DWGF)
    DWGO = utils.sumouter(goerr[:n],source[:n],out=DWGO)
    DWCI = utils.sumouter(cierr[:n],source[:n],out=DWCI)

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
        return np.array(self.state[:self.last_n])
    def init_weights(self,initial):
        "Initialize the weight matrices and derivatives"
        ni,ns,na = self.dims
        # gate weights
        for w in "WGI WGF WGO WCI".split():
            setattr(self,w,randu(ns,na)*initial)
            setattr(self,"D"+w,np.zeros((ns,na)))
        # peep weights
        for w in "WIP WFP WOP".split():
            setattr(self,w,randu(ns)*initial)
            setattr(self,"D"+w,np.zeros(ns))
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
            a = np.array(getattr(self,v))
            print(v, a.shape, np.amin(a), np.amax(a))
    def preSave(self):
        self.max_n = max(500,len(self.ci))
        self.allocate(1)
    def postLoad(self):
        self.allocate(getattr(self,"max_n",5000))
    def allocate(self,n):
        """Allocate space for the internal state variables.
        `n` is the maximum sequence length that can be processed."""
        ni,ns,na = self.dims
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        for v in vars.split():
            setattr(self,v,np.nan*np.ones((n,ns)))
        self.source = np.nan*np.ones((n,na))
        self.sourceerr = np.nan*np.ones((n,na))
    def reset(self,n):
        """Reset the contents of the internal state variables to `nan`"""
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        vars += " source sourceerr"
        for v in vars.split():
            getattr(self,v)[:,:] = np.nan
    def forward(self,xs):
        """Perform forward propagation of activations and update the
        internal state for a subsequent call to `backward`.
        Since this performs sequence classification, `xs` is a 2D
        array, with rows representing input vectors at each time step.
        Returns a 2D array whose rows represent output vectors for
        each input vector."""
        ni,ns,na = self.dims
        assert len(xs[0])==ni
        n = len(xs)
        self.last_n = n
        N = len(self.gi)
        if n>N: raise RecognitionError("input too large for LSTM model")
        self.reset(n)
        forward_py(n,N,ni,ns,na,xs,
                   self.source,
                   self.gix,self.gfx,self.gox,self.cix,
                   self.gi,self.gf,self.go,self.ci,
                   self.state,self.output,
                   self.WGI,self.WGF,self.WGO,self.WCI,
                   self.WIP,self.WFP,self.WOP)
        assert not np.isnan(self.output[:n]).any()
        return self.output[:n]
    def backward(self,deltas):
        """Perform backward propagation of deltas. Must be called after `forward`.
        Does not perform weight updating (for that, use the generic `update` method).
        Returns the `deltas` for the input vectors."""
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
    def walk(self):
        yield self
        for sub in self.nets:
            for x in sub.walk(): yield x
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
                self.dstats[i].append((np.amin(deltas),np.mean(deltas),np.amax(deltas)))
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
    def walk(self):
        yield self
        for x in self.net.walk(): yield x
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
    def walk(self):
        yield self
        for sub in self.nets:
            for x in sub.walk(): yield x
    def forward(self,xs):
        outputs = [net.forward(xs) for net in self.nets]
        outputs = zip(*outputs)
        outputs = [np.concatenate(l) for l in outputs]
        return outputs
    def backward(self,deltas):
        deltas = np.array(deltas)
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
        # states = [net.states() for net in self.nets] # FIXME
        outputs = zip(*outputs)
        outputs = [np.concatenate(l) for l in outputs]
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
    result = np.zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0] = 1.0
        result[2*i+1,j] = 1.0
    result[-1,0] = 1.0
    return result

def translate_back0(outputs,threshold=0.25):
    """Simple code for translating output from a classifier
    back into a list of classes. TODO/ATTENTION: this can
    probably be improved."""
    ms = np.amax(outputs,axis=1)
    cs = np.argmax(outputs,axis=1)
    cs[ms<threshold*np.amax(outputs)] = 0
    result = []
    for i in range(1,len(cs)):
        if cs[i]!=cs[i-1]:
            if cs[i]!=0:
                result.append(cs[i])
    return result

def translate_back(outputs,threshold=0.7,pos=0):
    """Translate back. Thresholds on class 0, then assigns the maximum class to
    each region. ``pos`` determines the depth of character information returned:
        * `pos=0`: Return list of recognized characters
        * `pos=1`: Return list of position-character tuples
        * `pos=2`: Return list of character-probability tuples
     """
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = np.tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,np.arange(1,np.amax(mask)+1))
    if pos==1: return maxima # include character position
    if pos==2: return [(c, outputs[r,c]) for (r,c) in maxima] # include character probabilities
    return [c for (r,c) in maxima] # only recognized characters

def log_mul(x,y):
    "Perform multiplication in the log domain (i.e., addition)."
    return x+y

def log_add(x,y):
    "Perform addition in the log domain."
    #return np.where(np.abs(x-y)>10,np.maximum(x,y),np.log(np.exp(x-y)+1)+y)
    return np.where(np.abs(x-y)>10,np.maximum(x,y),np.log(np.exp(np.clip(x-y,-20,20))+1)+y)

def forward_algorithm(match,skip=-5.0):
    """Apply the forward algorithm to an array of log state
    correspondence probabilities."""
    v = skip*np.arange(len(match[0]))
    result = []
    # This is a fairly straightforward dynamic programming problem and
    # implemented in close analogy to the edit distance:
    # we either stay in the same state at no extra cost or make a diagonal
    # step (transition into new state) at no extra cost; the only costs come
    # from how well the symbols match the network output.
    for i in range(0,len(match)):
        w = np.roll(v,1).copy()
        # extra cost for skipping initial symbols
        w[0] = skip*i
        # total cost is match cost of staying in same state
        # plus match cost of making a transition into the next state
        v = log_add(log_mul(v,match[i]),log_mul(w,match[i]))
        result.append(v)
    return np.array(result,'f')

def forwardbackward(lmatch):
    """Apply the forward-backward algorithm to an array of log state
    correspondence probabilities."""
    lr = forward_algorithm(lmatch)
    # backward is just forward applied to the reversed sequence
    rl = forward_algorithm(lmatch[::-1,::-1])[::-1,::-1]
    both = lr+rl
    return both

def ctc_align_targets(outputs,targets,threshold=100.0,verbose=0,debug=0,lo=1e-5):
    """Perform alignment between the `outputs` of a neural network
    classifier and some targets. The targets themselves are a time sequence
    of vectors, usually a unary representation of each target class (but
    possibly sequences of arbitrary posterior probability distributions
    represented as vectors)."""

    outputs = np.maximum(lo,outputs)
    outputs = outputs * 1.0/np.sum(outputs,axis=1)[:,np.newaxis]

    # first, we compute the match between the outputs and the targets
    # and put the result in the log domain
    match = np.dot(outputs,targets.T)
    lmatch = np.log(match)

    if debug:
        plt.figure("ctcalign"); plt.clf();
        plt.subplot(411); plt.imshow(outputs.T,interpolation='nearest',cmap=plt.cm.hot)
        plt.subplot(412); plt.imshow(lmatch.T,interpolation='nearest',cmap=plt.cm.hot)
    assert not np.isnan(lmatch).any()

    # Now, we compute a forward-backward algorithm over the matches between
    # the input and the output states.
    both = forwardbackward(lmatch)

    # We need posterior probabilities for the states, so we need to normalize
    # the output. Instead of keeping track of the normalization
    # factors, we just normalize the posterior distribution directly.
    epath = np.exp(both-np.amax(both))
    l = np.sum(epath,axis=0)[np.newaxis,:]
    epath /= np.where(l==0.0,1e-9,l)

    # The previous computation gives us an alignment between input time
    # and output sequence position as posteriors over states.
    # However, we actually want the posterior probability distribution over
    # output classes at each time step. This dot product gives
    # us that result. We renormalize again afterwards.
    aligned = np.maximum(lo,np.dot(epath,targets))
    l = np.sum(aligned,axis=1)[:,np.newaxis]
    aligned /= np.where(l==0.0,1e-9,l)

    if debug:
        plt.subplot(413); plt.imshow(epath.T,cmap=plt.cm.hot,interpolation='nearest')
        plt.subplot(414); plt.imshow(aligned.T,cmap=plt.cm.hot,interpolation='nearest')
        plt.ginput(1,0.01);
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
    def walk(self):
        for x in self.lstm.walk(): yield x
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
        self.outputs = np.array(self.lstm.forward(xs))
        return translate_back(self.outputs)
    def trainSequence(self,xs,cs,update=1,key=None):
        "Train with an integer sequence of codes."
        assert xs.shape[1]==self.Ni,"wrong image height"
        # forward step
        self.outputs = np.array(self.lstm.forward(xs))
        # CTC alignment
        self.targets = np.array(make_target(cs,self.No))
        self.aligned = np.array(ctc_align_targets(self.outputs,self.targets,debug=self.debug_align))
        # propagate the deltas back
        deltas = self.aligned-self.outputs
        self.lstm.backward(deltas)
        if update: self.lstm.update()
        # translate back into a sequence
        result = translate_back(self.outputs)
        # compute least square error
        self.error = np.sum(deltas**2)
        self.error_log.append(self.error**.5/len(cs))
        # compute class error
        self.cerror = levenshtein(cs,result)
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
        # tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in graphemelist(s)]
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

def getstates_for_display(net):
    """Get internal states of an LSTM network for making nice state
    plots. This only works on a few types of LSTM."""
    if isinstance(net,LSTM):
        return net.state[:net.last_n]
    if isinstance(net,Stacked) and isinstance(net.nets[0],LSTM):
        return net.nets[0].state[:net.nets[0].last_n]
    return None

