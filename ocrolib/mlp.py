from __future__ import with_statement

__all__ = "RectifierMLP".split()

import os,sys,os.path,re,math
import copy as pycopy
from random import sample as selection, shuffle, uniform
from numpy import *
from pylab import *
from scipy import *
# from utils import *
from native import *
import ocropy

def c_order(a):
    return tuple(a.strides)==tuple(sorted(a.strides,reverse=1))
def error(net,data,cls,subset=None):
    predicted = net.classify(data,subset=subset)
    if subset is not None:
        cls = take(cls,subset)
    assert predicted.shape==cls.shape,"wrong shape (predicted vs cls): %s != %s"%(predicted.shape,cls.shape)
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

double sigmoid(double x);
double max(double x,double y);
double H(double x);

inline double sigmoid(double x) {
    if(x<-200) x = -200;
    else if(x>200) x = 200;
    return 1.0/(1.0+exp(-x));
}

inline double max(double x,double y) {
    if(x>y) return x; else return y;
}

inline double H(double x) {
    if(x<=0) return 0; else return 1;
}

void forward(int n,int m,int l,float w1[m][n],float b1[m],float w2[l][m],float b2[l],
             int k,float data[k][n],float outputs[k][l]) {
    // printf("forward %d:%d:%d (%d)\n",n,m,l,k);
#pragma omp parallel for
    for(int row=0;row<k;row++) {
        float *x = data[row];
        float y[m];
        float *z = outputs[row];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j];
            y[i] = max(0.0,total);
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
    // printf("forward %d:%d:%d (%d)\n",n,m,l,k);
#pragma omp parallel for
    for(int row=0;row<k;row++) {
        float *x = data[row];
        float y[m];
        float z[l];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j];
            y[i] = max(0,total);
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
              int k,float data[k][n],int classes[k],float eta,int ntrain) {
    // printf("rectifier backward %d:%d:%d (%d)\n",n,m,l,k);
    assert(eta>=0.0);
    assert(eta<10.0);
#pragma omp parallel for
    for(int trial=0;trial<ntrain;trial++) {
        int row = (unsigned)(19.73*k*sin(trial))%k;
        // forward pass
        float *x = data[row];
        float y[m],z[l],delta2[l],delta1[m];
        for(int i=0;i<m;i++) {
            double total = b1[i];
            for(int j=0;j<n;j++) total += w1[i][j]*x[j];
            y[i] = max(0,total);
            assert(!isnan(y[i]));
        }
        for(int i=0;i<l;i++) {
            double total = b2[i];
            for(int j=0;j<m;j++) total += w2[i][j]*y[j];
            z[i] = sigmoid(total);
            assert(!isnan(z[i]));
        }
        if(trial%10000==0) {
            int count = 0;
            for(int i=0;i<m;i++) count += (y[i]==0.0);
            // printf("*** %d : %d zeros of %d\n",trial,count,m);
        }
        // backward pass
        int cls = classes[row];
        for(int i=0;i<l;i++) {
            delta2[i] = (z[i]-(i==cls)) * (z[i] * (1-z[i]));
        }
        for(int i=0;i<m;i++) {
            double total = 0.0;
            for(int j=0;j<l;j++)
                total += delta2[j] *  w2[j][i];
            total *= H(y[i]);
            delta1[i] = total;
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
''')

nnet_native.forward.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2F,A2F]
nnet_native.classify.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2F,A1I]
nnet_native.backward.argtypes = [I,I,I,A2F,A1F,A2F,A1F, I,A2F,A1I,F,I]

class RectifierMLP:
    def __init__(self):
        self.w1 = None
        self.verbose = 0
        self.eta = 0.1
        self.err = -1
    def check_finite(self):
        "Ensure all weights are finite."
        assert finite(self.w1)
        assert finite(self.b1)
        assert finite(self.w2)
        assert finite(self.b2)
    def init(self,data,cls,nhidden=None,eps=1e-2):
        data = data.reshape(len(data),prod(data.shape[1:]))
        scale = max(abs(amax(data)),abs(amin(data)))
        ninput = data.shape[1]
        if nhidden is None: nhidden = len(set(cls))
        noutput = amax(cls)+1
        # print ninput,nhidden,noutput
        self.w1 = array(data[selection(xrange(len(data)),nhidden)] * eps/scale,'f')
        self.b1 = array(uniform(-eps,eps,(nhidden,)),'f')
        self.w2 = array(uniform(-eps,eps,(noutput,nhidden)),'f')
        self.b2 = array(uniform(-eps,eps,(noutput,)),'f')
    def change_nhidden(self,data,cls,new_nhidden,subset=None):
        ninput,nhidden,noutput = self.shape()
        if nhidden==new_nhidden:
            pass
        elif nhidden>new_nhidden:
            keep = array([True]*nhidden)
            for i in selection(xrange(nhidden),nhidden-new_nhidden):
                keep[i] = False
            self.w1 = array(self.w1[keep,:],dtype='f',order="C")
            self.b1 = array(self.b1[keep],dtype='f',order="C")
            self.w2 = array(self.w2[:,keep],dtype='f',order="C")
        else:
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
            self.w2 = array(1.0*hstack([self.w2,scale*randn(len(self.w2),delta)]),dtype='f',order="C")
        assert c_order(self.w1)
        assert c_order(self.b1)
        assert c_order(self.w2)
        assert c_order(self.b2)
    def nhidden(self):
        assert self.w1.shape[0]==self.w2.shape[1]
        return self.w1.shape[0]
    def shape(self):
        assert self.w1.shape[0]==self.w2.shape[1]
        return self.w1.shape[1],self.w1.shape[0],self.w2.shape[0]
    def train(self,data,cls,etas=[(0.1,100000)]*30,nhidden=None,eps=1e-2,subset=None,verbose=0):
        data = data.reshape(len(data),prod(data.shape[1:]))
        if subset is not None:
            data = take(data,subset,axis=0)
            cls = take(cls,subset)
        if self.w1==None:
            self.init(data,cls,nhidden=nhidden,eps=eps)
        n,d = data.shape
        nc = self.w2.shape[0]
        for batch in range(len(etas)):
            eta,batchsize = etas[batch]
            if verbose: print "batch",batch,"(",eta,batchsize,")"
            self.eta = eta
            for trial in range(batchsize):
                if trial%10000==0: print trial
                index = randint(n)
                self.train1(data[index],unit(cls[index],nc))
            self.check_finite()
            err = error(self,data,cls)
            rate = err*1.0/len(data)
            if verbose: print "error",rate,err,len(data)
    def outputs(self,data,subset=None):
        "Compute posterior probabilities."
        data = data.reshape(len(data),prod(data.shape[1:]))
        assert data.shape[1]==self.w1.shape[1]
        if subset is not None:
            data = take(data,subset,axis=0)
            cls = take(cls,subset)
        return array([self.forward(v) for v in data])
    def classify(self,data,subset=None):
        "Classify the given input vector."
        data = data.reshape(len(data),prod(data.shape[1:]))
        assert data.shape[1]==self.w1.shape[1]
        if subset is not None:
            data = take(data,subset,axis=0)
        return array([argmax(self.forward(v)) for v in data])
    def forward(self,x):
        "Forward propagation step."
        assert len(x)==self.w1.shape[1]
        hidden = self.sigmoid(dot(self.w1,x) + self.b1)
        output = self.sigmoid(dot(self.w2,hidden) + self.b2)
        assert finite(output)
        return output
    def train1(self,x,target):
        "Backward propagation step."
        assert amin(x)>-10 and amax(x)<10
        eta = self.eta
        hidden,hderiv = self.dsigmoid(dot(self.w1,x) + self.b1)
        output,oderiv= self.dsigmoid(dot(self.w2,hidden) + self.b2)
        delta2 = (output-target) * oderiv
        delta1 = dot(delta2,self.w2).transpose() * hderiv
        self.w2 -= outer(eta*delta2,hidden)
        self.w1 -= outer(eta*delta1,x)
        self.b2 -= eta * delta2
        self.b1 -= eta * delta1
        return output
    def sigmoid(self,x):
        "Computes the nonlinearity."
        x = minimum(maximum(x,-200),200)
        return 1.0 / (1.0 + exp(-x))
    def dsigmoid(self,x):
        "Computes the nonlinearity and its derivative."
        y = self.sigmoid(x)
        return y,maximum(y * (1.0-y),sigmoid_floor)
    def set(self,other):
        "Set up a network with the given weights."
        self.w1 = 1*other.w1
        self.b1 = 1*other.b1
        self.w2 = 1*other.w2
        self.b2 = 1*other.b2
        self.eta = other.eta
        self.err = other.err
        assert c_order(self.w1)
        assert c_order(self.b1)
        assert c_order(self.w2)
        assert c_order(self.b2)
    def copy(self):
        "Clone the network"
        result = pycopy.copy(self)
        result.set(self)
        return result
    def save(self,stream):
        "Save the network to the stream."
        if isinstance(stream,basestring):
            with open(stream,"w") as stream: self.load(stream)
        self.w1.dump(stream)
        self.b1.dump(stream)
        self.w2.dump(stream)
        self.b2.dump(stream)
    def load(self,stream):
        "Load the network from the stream."
        if isinstance(stream,basestring):
            with open(stream) as stream: self.load(stream)
        self.w1 = load(stream)
        self.b1 = load(stream)
        self.w2 = load(stream)
        self.b2 = load(stream)
    def info(self):
        return tstr("shape",self.shape(),
                    "range",amin(self.w1),amax(self.w1),amin(self.w2),amax(self.w2))
    def train(self,data,cls,etas=[(0.1,100000)]*30,nhidden=None,eps=1e-2,subset=None,verbose=0):
        data = data.reshape(len(data),prod(data.shape[1:]))
        if subset is not None:
            data = take(data,subset,axis=0)
            cls = take(cls,subset)
        cls = array(cls,'i')
        if self.w1==None:
            self.init(data,cls,nhidden=nhidden,eps=eps)
        n,m,l = self.shape()
        for i in range(len(etas)):
            eta,batchsize = etas[i]
            if verbose: print "native batch",i,eta,batchsize
            assert data.dtype==dtype('f')
            assert cls.dtype==dtype('i')
            assert amin(data)>-100.0 and amax(data)<100
            assert amin(cls)>=0 and amax(cls)<10000
            assert eta>0.0 and eta<10.0
            assert type(batchsize)==int
            assert self.w1.flags["ALIGNED"]
            assert self.b1.flags["ALIGNED"]
            assert self.w2.flags["ALIGNED"]
            assert self.b2.flags["ALIGNED"]
            assert c_order(self.w1)
            assert c_order(self.b1)
            assert c_order(self.w2)
            assert c_order(self.b2)
            nnet_native.backward(n,m,l,self.w1,self.b1,self.w2,self.b2,
                                 len(data),data,cls,eta,batchsize)
            err = error(self,data,cls)
            rate = err*1.0/len(data)
            if verbose: print "error",rate,err,len(data)
    def outputs(self,data,subset=None):
        data = data.reshape(len(data),prod(data.shape[1:]))
        assert data.shape[1]==self.w1.shape[1]
        assert amin(data)>-100.0 and amax(data)<100
        if subset is not None:
            data = take(data,subset,axis=0)
            cls = take(cls,subset)
        result = zeros((len(data),self.w2.shape[0]),dtype='f')
        n,m,l = self.shape()
        nnet_native.forward(n,m,l,self.w1,self.b1,self.w2,self.b2,
                            len(data),data,result)
        return result
    def classify(self,data,subset=None):
        data = data.reshape(len(data),prod(data.shape[1:]))
        assert data.shape[1]==self.w1.shape[1]
        assert amin(data)>-100.0 and amax(data)<100
        if subset is not None:
            data = take(data,subset,axis=0)
        result = zeros(len(data),dtype='i')
        n,m,l = self.shape()
        nnet_native.classify(n,m,l,self.w1,self.b1,self.w2,self.b2,
                             len(data),data,result)
        return result

class NnModel:
    def __init__(self,nnet=None):
        self.nnet = nnet
        self.extractor = None
        self.nfeatures = None # number of features after extraction
        self.collect = [] # feature vectors for training
        self.values = [] # corresponding classes
        self.c2i = {}
        self.i2c = {}
    def preprocess(self,image):
        if self.extractor is None:
            return ocropy.as_numpy(image,flip=0)
        else:
            image = ocropy.as_narray(image,flip=0)
            out = ocropy.floatarray()
            self.extractor.extract(out,image)
            return ocropy.as_numpy(out,flip=0)
    def name(self):
        return "RectifierMlp"
    def cadd(self,image,c):
        image = ocropy.as_numpy(image,flip=1)
        preprocessed = self.preprocess(image).ravel()
        if self.nfeatures is not None: assert self.nfeatures==len(preprocessed)
        else: self.nfeatures = len(preprocessed)
        self.collect.append(preprocessed)
        self.values.append(c)
    def updateModel(self,etas=[(0.1,100000)]*15,verbose=1):
        self.i2c = sorted(list(set(self.values)))
        for i in range(len(self.i2c)): self.c2i[self.i2c[i]] = i
        self.data = array(self.collect)
        cls = array([self.c2i[v] for v in self.values],'i')
        self.collect = None
        self.nnet.train(self.data,cls,etas=etas,verbose=verbose)
    def coutputs(self,image,k=None):
        image = ocropy.as_numpy(image,flip=1)
        image = self.preprocess(image).ravel()
        v = self.nnet.outputs(image.reshape(1,len(image)))[0]
        return [(self.i2c[i],v[i]) for i in range(len(v))]
    def add(self,image,c):
        raise Exception("unimplemented")
    def outputs(self,image,k=None):
        raise Exception("unimplemented")
    def setExtractor(self,s):
        if type(s)==str:
            self.extractor = ocropy.make_IExtractor(s)
        else:
            self.extractor = s

class RectifierModel(NnModel):
    def __init__(self,*args,**kw):
        NnModel.__init__(self,RectifierMLP())
