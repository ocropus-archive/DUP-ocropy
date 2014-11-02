from pylab import *
from scipy.optimize.optimize import fmin_bfgs
from collections import Counter

###
### Helper classes.
###

class Err:
    def __init__(self,n=10000):
        self.n = n
        self.total = 0.0
        self.count = 0
    def add(self,x):
        l = 1.0/self.n
        self.total = (self.total*(1.0-l)+x*l)
        self.count += 1
    def value(self):
        return self.total

###
### Simple MLP implementation using gradient descent.
###

def sigmoid(x):
    return 1/(1+exp(-clip(x,-20,20)))
    
class MLP:
    def init(self,n,m,d):
        self.A = randn(n,m)
        self.a = randn(m)
        self.B = randn(m,d)
        self.b = randn(d)
    def forward(self,x):
        y = sigmoid(dot(self.A,x)+self.a)
        z = sigmoid(dot(self.B,y)+self.b)
        return z
    def backward(x,target,eta):
        y = sigmoid(dot(self.A,x)+self.a)
        z = sigmoid(dot(self.B,y)+self.b)
        delta_z = 2*(z-target)*z*(1-z)
        self.B -= eta * outer(delta_z,y)
        self.b -= eta * delta_z
        delta_y = dot(delta_z,self.B)*y*(1-y)
        self.A -= eta * outer(delta_y,x)
        self.a -= eta * delta_y

###
### Logistic regression using gradient descent.
###

def logreg_gd(data,target,eta=0.1,A=None,a=None,niter=None):
    n,m = data.shape
    n,d = target.shape
    if A is None: A = randn(d,m)
    if a is None: a = randn(d)
    e = Err(); d = Err()
    if niter is None: niter = maximum(3*len(data),1000000)
    for i in range(1,niter):
        j = i%n
        pred = sigmoid(dot(A,data[j].T)+a)
        delta = pred-target[j]
        e.add(argmax(pred)!=argmax(target[j]))
        d.add(sum(abs(pred-target[j])))
        delta = 2*delta*pred*(1-pred)
        l = eta * i**-0.5
        A -= l * outer(delta,data[j])
        a -= l * delta
        if i%100000==0: print i,e.value(),d.value()
    return (e.value(),A,a)

###
### Logistic regression using second order optimization methods.
###

def logpred(data,A):
    return sigmoid(dot(A,data.T)).T

def logloss(data,target,A,verbose=0):
    if A.ndim==1: A = A.reshape(target.shape[1],data.shape[1])
    # pred = logpred(data,A)
    loss = sum((logpred(data,A)-target)**2)
    if verbose: print "loss",loss
    return loss

def dlogloss(data,target,A):
    if A.ndim==1: A = A.reshape(target.shape[1],data.shape[1])
    pred = sigmoid(dot(A,data.T)).T
    delta = 2*(pred-target)*pred*(1-pred)
    result = dot(delta.T,data)
    return result.ravel()

def logreg_opt(data,targets,start=None,maxiter=100000):
    """Logistic regression using second order optimization methods."""
    n,d = data1.shape
    n,c = targets.shape
    data = c_[data,ones(len(data))]
    A = start
    if A is None: A = 0.01*randn(c,d+1)
    def f(x): return logloss(data,targets,x,verbose=1)
    def fprime(x): return dlogloss(data,targets,x)
    result = fmin_bfgs(f,A.ravel(),fprime=fprime,maxiter=maxiter)
    result.shape = (c,d+1)
    return result

###
### Linear regression with square loss and L2 regularization.
###

def lstsq_l2(data,targets,l=0.0):
    """Naive implementation of ridge regression (regularlized least square),
    using the formula M = (X^T X + l I)^-1 X^T Y"""
    n,m = data.shape
    p = dot(linalg.inv(dot(data.T,data)+diag(l*ones(m))),data.T)
    result = dot(p,targets)
    # print "lstsq_l2",p.shape,targets.shape,result.shape
    return result

###
### Logistic regression based on EM and linear regression.  Optionally,
### can use L2 regularization.
###

def logit(p):
    return log(p)-log(1-p)

def logreg_fp(data,targets,lstsq=linalg.lstsq,eta=10.0,rtol=1e-3,maxiter=1000,verbose=0,miniter=10,tol=5.0,initial=None):
    """Logistic regression by fixed point iterations.  By default uses
    an unregularized least square solver, other least square solvers can be
    used as well."""
    if initial is None:
        ntargets = targets
    else:
        ntargets = logit(initial)+(targets-initial)*eta
    last = inf
    for i in range(maxiter):
        A = lstsq(data,ntargets)
        if type(A)==tuple: A = A[0]
        pred = dot(data,A)
        if i==0 and verbose:
            print "lerror",sum((pred-targets)**2)
        spred = sigmoid(pred)
        deltas = (targets-spred)
        ntargets = pred + deltas*eta
        error = sum((spred-targets)**2)
        improvement = (last-error)/error
        if verbose:
            print i,error,improvement,
            if verbose>1: "/",mean(A),sqrt(var(A)),"/",mean(deltas),sqrt(var(deltas)),
            print
        if i>miniter and (improvement<rtol or (last-error)<tol): break
        last = error
    return A

def logreg_l2_fp(data,targets,l,**kw):
    """Logistic regression with L2 regularization.  This uses the fp solver
    above, and passes an L2 linear solver as a subroutine."""
    def lstsq(data,targets):
        return lstsq_l2(data,targets,l=l)
    return logreg_fp(data,targets,lstsq=lstsq,**kw)

class LinClassifier:
    def train(self,data,classes,k=10,linear=0,l=1e-4):
        assert data.ndim>=2
        assert classes.ndim==1
        assert len(data)==len(classes)
        assert len(set(classes))<200
        self.classlist = [c for c,n in Counter(classes).most_common(200)]
        targets = array([classes==c for c in self.classlist],'i').T
        ys = make2d(data)
        if linear:
            ys = c_[ones(len(ys)),ys]
            M2 = linalg.lstsq(ys,targets)[0]
        else:
            ys = c_[ones(len(ys)),ys]
            M2 = logreg_l2_fp(ys,targets,l=l)
        b = M2[0,:]
        M = M2[1:,:]
        self.R = M
        self.r = b
        self.linear = linear
    def outputs(self,data):
        assert data.ndim>=2
        if self.linear:
            pred = dot(make2d(data),self.R)+self.r[newaxis,:]
        else:
            pred = sigmoid(dot(make2d(data),self.R)+self.r[newaxis,:])
        return [[(c,p[i]) for i,c in enumerate(self.classlist)] for j,p in enumerate(pred)]
    def classify(self,data):
        assert data.ndim>=2
        pred = argmax(dot(make2d(data),self.R)+self.r[newaxis,:],axis=1)
        return [self.classlist[p] for p in pred]

###
### Linear regression with square loss and L1 regularization.
###

def lstsq_l1(data,targets,l=0.0):
    assert False

def pca(data,k=5,frac=0.99,whiten=0):
    """Computes a PCA and a whitening.  The number of
    components can be specified either directly or as a fraction
    of the total sum of the eigenvalues.  The function returns
    the transformed data, the mean, the eigenvalues, and 
    the eigenvectors."""
    n,d = data.shape
    mean = average(data,axis=0).reshape(1,d)
    data = data - mean.reshape(1,d)
    cov = dot(data.T,data)/n
    evals,evecs = linalg.eigh(cov)
    top = argsort(-evals)
    evals = evals[top[:k]]
    evecs = evecs.T[top[:k]]
    assert evecs.shape==(k,d)
    ys = dot(evecs,data.T)
    assert ys.shape==(k,n)
    if whiten: ys = dot(diag(sqrt(1.0/evals)),ys)
    return (ys.T,mean,evals,evecs)

def make2d(data):
    """Convert any input array into a 2D array by flattening axes 1 and over."""
    if data.ndim==1: return array([data])
    if data.ndim==2: return data
    return data.reshape(data.shape[0],prod(data.shape[1:]))

class LinPcaClassifier:
    def train(self,data,classes,k=10,linear=0,l=1e-4,classlist=None):
        assert data.ndim>=2
        assert classes.ndim==1
        assert len(data)==len(classes)
        assert len(set(classes))<200
        if classlist is None:
            self.classlist = [c for c,n in Counter(classes).most_common(200)]
        else:
            self.classlist = classlist
        targets = array([classes==c for c in self.classlist],'i').T
        (ys,mu,vs,tr) = pca(make2d(data),k=k)
        if linear:
            ys = c_[ones(len(ys)),ys]
            M2 = linalg.lstsq(ys,targets)[0]
        else:
            ys = c_[ones(len(ys)),ys]
            M2 = logreg_l2_fp(ys,targets,l=l)
        b = M2[0,:]
        M = M2[1:,:]
        self.R = dot(M.T,tr)
        self.r = b-dot(self.R,mu.ravel())
        self.linear = linear
    def outputs(self,data):
        assert data.ndim>=2
        if self.linear:
            pred = dot(make2d(data),self.R.T)+self.r[newaxis,:]
        else:
            pred = sigmoid(dot(make2d(data),self.R.T)+self.r[newaxis,:])
        return [[(c,p[i]) for i,c in enumerate(self.classlist)] for j,p in enumerate(pred)]
    def classify(self,data):
        assert data.ndim>=2
        pred = argmax(dot(make2d(data),self.R.T)+self.r[newaxis,:],axis=1)
        return [self.classlist[p] for p in pred]

from scipy.spatial.distance import cdist

class LinKernelClassifier:
    def train(self,rdata,classes,rprotos,sigma,linear=0,l=0.0):
        global data,dists,protos
        data = make2d(rdata)
        protos = make2d(rprotos)
        print "training",data.shape,protos.shape,sigma,Counter(classes).most_common(5)
        assert data.ndim>=2
        assert classes.ndim==1
        assert protos.shape[1]==data.shape[1],\
            "data shape %s != protos shape %s"%(data.shape[1:],protos.shape[1:])
        assert len(data)==len(classes)
        assert len(set(classes))<200
        self.classlist = [c for c,n in Counter(classes).most_common(200)]
        dists = cdist(data,protos,'euclidean')
        order = argsort(dists[:,0])
        dists = dists[order]
        data = data[order]
        classes = classes[order]
        print dists.shape
        mdists = mean(dists,axis=0)
        print mdists
        ys = c_[ones(len(data)),dists]
        targets = array([classes==c for c in self.classlist],'i').T
        if linear:
            M2 = linalg.lstsq(ys,targets)[0]
        else:
            M2 = logreg_l2_fp(ys,targets,l=l)
        self.protos = protos
        self.M = M2
        self.linear = linear
    def outputs(self,data):
        assert data.ndim>=2
        data = make2d(data)
        # ys = c_[ones(len(data)),cdist(data,self.protos,'euclidean')]
        if self.linear:
            pred = dot(make2d(data),self.M.T)
        else:
            pred = sigmoid(make2d(data),self.M.T)
        return [[(c,p[i]) for i,c in enumerate(self.classlist)] for j,p in enumerate(pred)]
    def classify(self,data):
        assert data.ndim>=2
        data = make2d(data)
        ys = c_[ones(len(data)),cdist(make2d(data),self.protos,'euclidean')]
        pred = argmax(dot(ys,self.M),axis=1)
        return [self.classlist[p] for p in pred]
    
###
### simple density model
###

class DiagGaussian:
    def __init__(self,data,default_sigma=0.1):
        data = make2d(data)
        default = default_sigma*ones(data[0].size)
        self.n = len(data)
        if len(data)<1:
            self.mu = None
            self.sigmas = None
        elif len(data)<2:
            self.mu = mean(data,axis=0)
            self.sigmas = default
            assert self.mu.size==self.sigmas.size
        else:
            self.mu = mean(data,axis=0)
            l = 1.0/len(data)
            self.sigmas = l*default + (1-l)*sqrt(var(data,axis=0))
            assert self.mu.size==self.sigmas.size
    def cost(self,v):
        if self.mu is None: return inf
        return norm((v-self.mu)/2.0/self.sigmas)
    def costs(self,vs):
        if self.mu is None: return [inf]*len(vs)
        return array([self.cost(v) for v in vs])
