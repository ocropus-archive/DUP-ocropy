from __future__ import with_statement

__all__ = "MLP".split()

import os,sys,os.path,re,math
import copy as pycopy
from random import sample as selection, shuffle, uniform
from numpy import *
from pylab import *
from scipy import *

import common
ocrolib = common
from native import *

def c_order(a):
    return tuple(a.strides)==tuple(sorted(a.strides,reverse=1))

def error(net,data,cls,subset=None):
    predicted = net.classify(data,subset=subset)
    if subset is not None:
        cls = take(cls,subset)
    assert predicted.shape==cls.shape,\
        "wrong shape (predicted vs cls): %s != %s"%(predicted.shape,cls.shape)
    return sum(cls!=predicted)

def finite(x):
    "Make sure that all entries of x are finite."
    return not isnan(x).any() and not isinf(x).any()

verbose_examples = 0
sigmoid_floor = 0.0

nnet_native = compile_and_load(r'''
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int verbose = 0;

double sigmoid(double x);
double max(double x,double y);

inline double sigmoid(double x) {
    if(x<-200) x = -200;
    else if(x>200) x = 200;
    return 1.0/(1.0+exp(-x));
}

inline double max(double x,double y) {
    if(x>y) return x; else return y;
}

void forward(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
             int k,float data[k][n],float outputs[k][l]) {
    if(verbose) printf("forward %d:%d:%d (%d)\n",n,m,l,k);
#pragma omp parallel for
    for(int row=0;row<k;row++) {
        float *x = data[row];
        float y[m];
        float *z = outputs[row];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j];
            y[i] = sigmoid(total);
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
        }
    }
}

int argmax(int k,float z[k]) {
    int mi = 0;
    float mv = z[0];
    for(int i=1;i<k;i++) {
        if(z[i]<mv) continue;
        mv = z[i];
        mi = i;
    }
    return mi;
}

void classify(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
             int k,float data[k][n],int classes[k]) {
    if(verbose) printf("classify %d:%d:%d (%d)\n",n,m,l,k);
#pragma omp parallel for
    for(int row=0;row<k;row++) {
        float *x = data[row];
        float y[m];
        float z[l];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j];
            y[i] = sigmoid(total);
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
        }
        classes[row] = argmax(l,z);
    }
}

void backward(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
              int k,float data[k][n],int classes[k],float eta,int ntrain,
              int nsamples,int samples[nsamples]) {
    if(verbose) printf("backward %d:%d:%d (%d)\n",n,m,l,k);
    assert(eta>0.0);
    assert(eta<10.0);
#pragma omp parallel for
    for(int trial=0;trial<ntrain;trial++) {
        int row;
        if(nsamples>0) row = samples[(unsigned)(19.73*k*sin(trial))%nsamples];
        else row = (unsigned)(19.73*k*sin(trial))%k;
        // forward pass
        float *x = data[row];
        float y[m],z[l],delta2[l],delta1[m];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j];
            y[i] = sigmoid(total);
            assert(!isnan(y[i]));
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
            assert(!isnan(z[i]));
        }
        // backward pass
        int cls = classes[row];
        for(int i=0;i<l;i++) {
            double total = (z[i]-(i==cls));
            delta2[i] =  total * z[i] * (1-z[i]);
        }
        for(int i=0;i<m;i++) {
            double total = 0.0;
            for(int j=0;j<l;j++)
                total += delta2[j] *  w2[j][i];
            delta1[i] = total * y[i] * (1-y[i]);
        }
        for(int i=0;i<l;i++) {
            for(int j=0;j<m;j++) {
                w2[i][j] -= eta*delta2[i]*y[j];
            }
        }
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                w1[i][j] -= eta*delta1[i]*x[j];
            }
        }
    }
}

typedef signed char byte;
#define BSCALE 100.0

void forward_b(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
               int k,byte data[k][n],float outputs[k][l]) {
    if(verbose) printf("forward %d:%d:%d (%d)\n",n,m,l,k);
#pragma omp parallel for
    for(int row=0;row<k;row++) {
        byte *x = data[row];
        float y[m];
        float *z = outputs[row];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j]/BSCALE;
            y[i] = sigmoid(total);
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
        }
    }
}

void classify_b(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
                int k,byte data[k][n],int classes[k]) {
    if(verbose) printf("classify %d:%d:%d (%d)\n",n,m,l,k);
#pragma omp parallel for
    for(int row=0;row<k;row++) {
        byte *x = data[row];
        float y[m];
        float z[l];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j]/BSCALE;
            y[i] = sigmoid(total);
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
        }
        classes[row] = argmax(l,z);
    }
}

void backward_b(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
                int k,byte data[k][n],int classes[k],float eta,int ntrain,
                int nsamples,int samples[nsamples]) {
    if(verbose) printf("backward %d:%d:%d (%d)\n",n,m,l,k);
    assert(eta>0.0);
    assert(eta<10.0);
#pragma omp parallel for
    for(int trial=0;trial<ntrain;trial++) {
        int row;
        if(nsamples>0) row = samples[(unsigned)(19.73*k*sin(trial))%nsamples];
        else row = (unsigned)(19.73*k*sin(trial))%k;
        // forward pass
        byte *x = data[row];
        float y[m],z[l],delta2[l],delta1[m];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j]/BSCALE;
            y[i] = sigmoid(total);
            assert(!isnan(y[i]));
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
            assert(!isnan(z[i]));
        }
        // backward pass
        int cls = classes[row];
        for(int i=0;i<l;i++) {
            double total = (z[i]-(i==cls));
            delta2[i] =  total * z[i] * (1-z[i]);
        }
        for(int i=0;i<m;i++) {
            double total = 0.0;
            for(int j=0;j<l;j++)
                total += delta2[j] *  w2[j][i];
            delta1[i] = total * y[i] * (1-y[i]);
        }
        for(int i=0;i<l;i++) {
            for(int j=0;j<m;j++) {
                w2[i][j] -= eta*delta2[i]*y[j];
            }
        }
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                w1[i][j] -= eta*delta1[i]*x[j]/BSCALE;
            }
        }
    }
}
''')

nnet_native.forward.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2F,A2F]
nnet_native.classify.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2F,A1I]
nnet_native.backward.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2F,A1I,F,I]

nnet_native.forward_b.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2B,A2F]
nnet_native.classify_b.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2B,A1I]
nnet_native.backward_b.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2B,A1I,F,I,I,A1I]


class MLP:
    def __init__(self):
        self.w1 = None
        self.verbose = 0
        self.eta = 0.1
        self.err = -1
    def copy(self):
        mlp = MLP()
        mlp.w1 = self.w1.copy()
        mlp.w2 = self.w2.copy()
        mlp.b1 = self.b1.copy()
        mlp.b2 = self.b2.copy()
        mlp.verbose = self.verbose
        mlp.err = -1
        mlp.eta = self.eta
        return mlp
    def checkWeightShape(self):
        """Ensure that the internal weights have the right shape and
        alignment for passing them to native code."""
        assert c_order(self.w1)
        assert c_order(self.b1)
        assert c_order(self.w2)
        assert c_order(self.b2)
        assert self.w1.flags["ALIGNED"]
        assert self.b1.flags["ALIGNED"]
        assert self.w2.flags["ALIGNED"]
        assert self.b2.flags["ALIGNED"]
        assert self.w1.shape[0]==self.w2.shape[1]
    def init(self,data,cls,nhidden=None,eps=1e-2):
        data = data.reshape(len(data),prod(data.shape[1:]))
        scale = max(abs(amax(data)),abs(amin(data)))
        ninput = data.shape[1]
        if nhidden is None: nhidden = len(set(cls))
        noutput = amax(cls)+1
        self.w1 = array(data[selection(xrange(len(data)),nhidden)] * eps/scale,'f')
        self.b1 = array(uniform(-eps,eps,(nhidden,)),'f')
        self.w2 = array(uniform(-eps,eps,(noutput,nhidden)),'f')
        self.b2 = array(uniform(-eps,eps,(noutput,)),'f')
    def decreaseHidden(self,data,cls,new_nhidden):
        ninput,nhidden,noutput = self.shape()
        keep = array([True]*nhidden)
        for i in selection(xrange(nhidden),nhidden-new_nhidden):
            keep[i] = False
        self.w1 = array(self.w1[keep,:],dtype='f',order="C")
        self.b1 = array(self.b1[keep],dtype='f',order="C")
        self.w2 = array(self.w2[:,keep],dtype='f',order="C")
    def increaseHidden(self,data,cls,new_nhidden):
        vs = []
        bs = []
        delta = new_nhidden-nhidden
        for i in range(delta):
            a,b = selection(xrange(nhidden),2)
            l = 0.8*rand(1)[0]+0.1
            v = l*self.w1[a] + (1-l)*self.w1[b]
            vs.append(v)
            b = l*self.b1[a] + (1-l)*self.b1[b]
            bs.append(b)
        self.w1 = array(1.0*vstack([self.w1,array(vs)]),dtype='f',order="C")
        self.b1 = array(1.0*hstack([self.b1,array(bs)]),dtype='f',order="C")
        scale = 0.01*mean(abs(self.w2))
        vecs = [self.w2,scale*randn(len(self.w2),delta)]
        self.w2 = array(1.0*hstack(vecs),dtype='f',order="C")
    def changeHidden(self,data,cls,new_nhidden,subset=None):
        if self.nhidden()==new_nhidden: return
        elif self.nhidden()>new_nhidden: self.decreaseHidden(data,cls,new_nhidden)
        else: self.increaseHidden(data,cls,new_nhidden)
        self.checkWeightShape()
    def nhidden(self):
        return self.w1.shape[0]
    def shape(self):
        assert self.w1.shape[0]==self.w2.shape[1]
        return self.w1.shape[1],self.w1.shape[0],self.w2.shape[0]
    def train(self,data,cls,etas=[(0.1,100000)]*30,
              nhidden=None,eps=1e-2,subset=None,verbose=1,samples=None):
        data = data.reshape(len(data),prod(data.shape[1:]))
        if subset is not None:
            data = take(data,subset,axis=0)
            cls = take(cls,subset)
        cls = array(cls,'i')
        if self.w1==None:
            self.init(data,cls,nhidden=nhidden,eps=eps)
        if verbose:
            err = error(self,data,cls)
            rate = err*1.0/len(data)
            print "starting",data.shape,data.dtype
            print "error",rate,err,len(data)
            print "ranges",amin(self.w1),amax(self.w1),amin(self.w2),amax(self.w2)
        n,m,l = self.shape()
        for i in range(len(etas)):
            eta,batchsize = etas[i]
            if verbose: print "native batch",i,eta,batchsize
            assert cls.dtype==dtype('i')
            assert amin(cls)>=0 and amax(cls)<10000
            assert eta>0.0 and eta<10.0
            assert type(batchsize)==int
            self.checkWeightShape()
            if samples is None: samples = array([],'i')
            assert samples.dtype==dtype('i')
            if data.dtype==dtype('f'):
                assert amin(data)>-100.0 and amax(data)<100
                nnet_native.backward(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                     len(data),data,cls,eta,batchsize,
                                     len(samples),samples)
            elif data.dtype==dtype('int8'):
                nnet_native.backward_b(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                       len(data),data,cls,eta,batchsize,
                                       len(samples),samples)
            else:
                raise Exception("data has unknown type")
            err = error(self,data,cls)
            rate = err*1.0/len(data)
            if verbose:
                print "error",rate,err,len(data)
                print "ranges",amin(self.w1),amax(self.w1),amin(self.w2),amax(self.w2)
            self.error_rate = rate
    def outputs(self,data,subset=None):
        data = data.reshape(len(data),prod(data.shape[1:]))
        assert data.shape[1]==self.w1.shape[1]
        if subset is not None:
            data = take(data,subset,axis=0)
            cls = take(cls,subset)
        result = zeros((len(data),self.w2.shape[0]),dtype='f')
        n,m,l = self.shape()
        if data.dtype==dtype('f'):
            assert amin(data)>-100.0 and amax(data)<100
            nnet_native.forward(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                len(data),data,result)
        elif data.dtype==dtype('int8'):
            nnet_native.forward_b(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                  len(data),data,result)
        else:
            raise Exception("data has unknown type")
        return result
    def classify(self,data,subset=None):
        data = data.reshape(len(data),prod(data.shape[1:]))
        assert data.shape[1]==self.w1.shape[1]
        if subset is not None:
            data = take(data,subset,axis=0)
        result = zeros(len(data),dtype='i')
        n,m,l = self.shape()
        if data.dtype==dtype('f'):
            assert amin(data)>-100.0 and amax(data)<100
            nnet_native.classify(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                 len(data),data,result)
        elif data.dtype==dtype('int8'):
            nnet_native.classify_b(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                   len(data),data,result)
        else:
            raise Exception("data has unknown type")
        return result

class AutoMLP(MLP):
    def __init__(self):
        self.verbose = 1
        self.log_eta_var = 0.2
        self.log_nh_var = 0.2
        self.nhidden = [10,20,40,80]
        self.eta = 0.1
        self.min_round = 100000
        self.max_round = 10000000
        self.epochs_per_round = 5
        self.max_rounds = 16
        self.max_pool = 4
    def train(self,data,classes,verbose=0):
        n = len(data)
        testing = array(selection(xrange(n),n/10),'i')
        training = setdiff1d(array(xrange(n),'i'),testing)
        testset = data[testing,:]
        testclasses = classes[testing]
        ntrain = max(self.min_round,len(data)*self.epochs_per_round)
        pool = []
        for nh in self.nhidden:
            mlp = MLP()
            mlp.eta = self.eta
            mlp.train(data,classes,etas=[(self.eta,ntrain)],
                      nhidden=nh,
                      verbose=0,
                      samples=training)
            mlp.err = error(mlp,testset,testclasses)
            if verbose: print "AutoMLP initial",mlp.eta,nh,mlp.err
            pool.append(mlp)
        for i in range(self.max_rounds):
            mlp = selection(pool,1)[0]
            mlp = mlp.copy()
            new_eta = exp(log(mlp.eta)+randn()*self.log_eta_var)
            new_nh = max(2,int(log(mlp.nhidden())+randn()*self.log_nh_var))
            # print "eta",mlp.eta,new_eta,"nh",mlp.nhidden(),new_nh
            mlp.eta = new_eta
            mlp.changeHidden(data,classes,new_nh)
            mlp.train(data,classes,etas=[(self.eta,ntrain)],
                      verbose=(self.verbose>1),samples=training)
            mlp.err = error(mlp,testset,testclasses)
            print "AutoMLP pool",mlp.err,"(%.3f,%d)"%(mlp.eta,mlp.nhidden()),\
                [x.err for x in pool]
            pool += [mlp]
            errs = [x.err for x in pool]
            if len(errs)>self.max_pool:
                choice = argsort(errs)
                pool = list(take(pool,choice[:self.max_pool]))
        best = argmin([mlp.err+0.1*mlp.nhidden() for mlp in pool])
        mlp = pool[best]
        self.assign(mlp)
    def assign(self,mlp):
        for k,v in mlp.__dict__.items():
            if k[0]=="_": continue
            setattr(self,k,v)

def test():
    data = array(randn(10000,2),'f')
    data = array(2*(data>0)-1,'f')
    data += 0.1*randn(10000,2)
    classes = array(data[:,0]*data[:,1]>0,'i')
    data += 0.4
    bdata = array(100.0*clip(data,-1.1,1.1),'int8')
    mlp = AutoMLP()
    mlp.max_rounds = 32
    mlp.train(bdata[:9000,:],classes[:9000],verbose=1)
    pred = mlp.classify(data[9000:])
    print sum(pred!=classes[9000:])
    print mlp.w1.shape,mlp.w2.shape

class MlpModel(common.ClassifierModel):
    makeClassifier = MLP
    makeExtractor = ocrolib.ScaledFE
    def __init__(self):
        common.ClassifierModel.__init__(self)
    def name(self):
        return str(self)
    def setExtractor(self,e):
        pass

class AutoMlpModel(common.ClassifierModel):
    makeClassifier = AutoMLP
    makeExtractor = ocrolib.ScaledFE
    def __init__(self):
        common.ClassifierModel.__init__(self)
    def name(self):
        return str(self)
    def setExtractor(self,e):
        pass
