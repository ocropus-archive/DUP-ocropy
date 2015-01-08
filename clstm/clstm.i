// -*- C++ -*-

%{
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wuninitialized"
%}

%module(docstring="C-version of the ocropy LSTM implementation") clstm;
%feature("autodoc",1);
%include "typemaps.i"
%include "std_string.i"
%include "std_shared_ptr.i"
%shared_ptr(INetwork)
#ifdef SWIGPYTHON
%include "cstring.i"
#endif

%{
#include <memory>
#include "clstm.h"
using namespace ocropus;
using namespace std;
%}

typedef float Float;
using std::string;

#ifdef SWIGPYTHON
%exception {
    try {
        $action
    }
    catch(const char *s) {
        PyErr_SetString(PyExc_IndexError,s);
        return NULL;
    }
    catch(...) {
        PyErr_SetString(PyExc_IndexError,"unknown exception in iulib");
        return NULL;
    }
}
#endif

%{
#include "numpy/arrayobject.h"
%}

%init %{
import_array();
%}

/* create simple interface definitions for the built-in Sequence and Vec types */

struct Classes {
    Classes();
    ~Classes();
    %rename(__getitem__) operator[];
    int operator[](int i);
    int size();
    void resize(int);
};
%extend Classes {
    void __setitem__(int i,int value) {
        (*$self)[i] = value;
    }
}

struct Vec {
    Vec();
    Vec(int);
    %rename(__getitem__) operator[];
    float operator[](int i);
    int size();
};
%extend Vec {
    void __setitem__(int i,float value) {
        (*$self)[i] = value;
    }
}

struct Mat {
    Mat();
    Mat(int,int);
    %rename(__getitem__) operator();
    float operator()(int i,int j);
    int rows();
    int cols();
};
%extend Mat {
    void setValue(int i,int j,float value) {
        (*$self)(i,j) = value;
    }
    void set(PyObject *object_) {
        Mat *a = $self;
        if(!object_) throw "null pointer";
        if(!PyArray_Check(object_)) throw "expectd a numpy array";
        PyArrayObject *obj = (PyArrayObject *)object_;
        if((obj->flags&NPY_CONTIGUOUS)==0) {
            obj = (PyArrayObject*)PyArray_ContiguousFromObject(object_,obj->descr->type_num,1,4);
            if(!obj) throw "contiguous conversion failed";
        }
        int rank = PyArray_NDIM(obj);
        if(rank!=2) throw "rank must be 2";
        int N = PyArray_DIM(obj,0);
        int d = PyArray_DIM(obj,1);
        a->resize(N,d);
        int t = obj->descr->type_num;
        if(t==PyArray_FLOAT) {
            float *data = (float*)PyArray_DATA(obj);
            for(int t=0;t<N;t++) {
                for(int i=0;i<d;i++) (*a)(t,i) = data[t*d+i];
            }
        } else {
            throw "numpy array must be float32 type";
        }
        if((PyObject*)obj!=object_) Py_DECREF(obj);
    }
    void get(PyObject *object_) {
        Mat *a = $self;
        if(!object_) throw "null pointer";
        if(!PyArray_Check(object_)) throw "expected a numpy array";
        PyArrayObject *obj = (PyArrayObject *)object_;
        int rank = PyArray_NDIM(obj);
        if(rank!=2) throw "rank must be 2";
        int N = PyArray_DIM(obj,0);
        if(N!=a->rows()) throw "size mismatch (N)";
        int d = PyArray_DIM(obj,1);
        if(d!=a->cols()) throw "size mismatch (d)";
        if((obj->flags&NPY_CONTIGUOUS)==0)
            throw "output array is not contiguous";
        int t = obj->descr->type_num;
        if(t==PyArray_FLOAT) {
            float *data = (float*)PyArray_DATA(obj);
            for(int t=0;t<N;t++) {
                for(int i=0;i<d;i++) data[t*d+i] = (*a)(t,i);
            }
        } else {
            throw "numpy array must be float32 type";
        }
    }
}

struct Sequence {
    Sequence();
    ~Sequence();
    int size();
    %rename(__getitem__) operator[];
    Vec &operator[](int i);
};
%extend Sequence {
    int length() {
        return $self->size();
    }
    int depth() {
        if($self->size()==0) return -1;
        return (*$self)[0].size();
    }
    void assign(Sequence &other) {
        $self->resize(other.size());
        for(int t=0;t<$self->size();t++)
            (*$self)[t] = other[t];
    }
    void set(PyObject *object_) {
        Sequence *a = $self;
        if(!object_) throw "null pointer";
        if(!PyArray_Check(object_)) throw "expectd a numpy array";
        PyArrayObject *obj = (PyArrayObject *)object_;
        if((obj->flags&NPY_CONTIGUOUS)==0) {
            obj = (PyArrayObject*)PyArray_ContiguousFromObject(object_,obj->descr->type_num,1,4);
            if(!obj) throw "contiguous conversion failed";
        }
        int rank = PyArray_NDIM(obj);
        if(rank!=2) throw "rank must be 2";
        int N = PyArray_DIM(obj,0);
        int d = PyArray_DIM(obj,1);
        a->resize(N);
        int t = obj->descr->type_num;
        if(t==PyArray_FLOAT) {
            float *data = (float*)PyArray_DATA(obj);
            for(int t=0;t<N;t++) {
                (*a)[t].resize(d);
                for(int i=0;i<d;i++) (*a)[t][i] = data[t*d+i];
            }
        } else {
            throw "numpy array must be float32 type";
        }
        if((PyObject*)obj!=object_) Py_DECREF(obj);
    }
    void get(PyObject *object_) {
        Sequence *a = $self;
        if(!object_) throw "null pointer";
        if(!PyArray_Check(object_)) throw "expected a numpy array";
        PyArrayObject *obj = (PyArrayObject *)object_;
        int rank = PyArray_NDIM(obj);
        if(rank!=2) throw "rank must be 2";
        int N = PyArray_DIM(obj,0);
        if(N!=a->size()) throw "size mismatch (N)";
        int d = PyArray_DIM(obj,1);
        for(int t=0;t<N;t++) if((*a)[t].size()!=d) throw "size mismatch (d)";
        if((obj->flags&NPY_CONTIGUOUS)==0)
            throw "output array is not contiguous";
        int t = obj->descr->type_num;
        if(t==PyArray_FLOAT) {
            float *data = (float*)PyArray_DATA(obj);
            for(int t=0;t<N;t++) {
                for(int i=0;i<d;i++) data[t*d+i] = (*a)[t][i];
            }
        } else {
            throw "numpy array must be float32 type";
        }
    }
}

struct INetwork {
    virtual ~INetwork() = 0;
    Float softmax_floor = 1e-5;
    bool softmax_accel = false;
    // Float lr = 1e-4;
    // Float momentum = 0.9;
    Sequence inputs,d_inputs;
    Sequence outputs,d_outputs;
    virtual int ninput();
    virtual int noutput();
    virtual void init(int no,int ni);
    virtual void init(int no,int nh,int ni);
    virtual void init(int no,int nh2,int nh,int ni);
    virtual void forward();
    virtual void backward();
    void info(string prefix);
    void train(Sequence &xs,Sequence &targets);
    void ctrain(Sequence &xs,Classes &cs);
    void ctrain_accelerated(Sequence &xs,Classes &cs,Float lo=1e-5);
    void cpred(Classes &preds,Sequence &xs);
    void setLearningRate(Float, Float);
    void setInputs(Sequence &inputs);
    void setTargets(Sequence &targets);
    void setClasses(Classes &classes);
    // typedef function<void (const string &,Eigen::Ref<Mat>,Eigen::Ref<Mat>)> WeightFun;
    // typedef function<void (const string &,Sequence *)> StateFun;
    // void weights(const string &prefix,WeightFun f);
    // void states(const string &prefix,StateFun f);
    Sequence *getState(string name);
    void add(shared_ptr<INetwork> net);
    void save(const char *fname);
    void load(const char *fname);
};
%extend INetwork {
    void add(INetwork *net) {
        $self->add(shared_ptr<INetwork>(net));
    }
    void setAttr(string key,string value) {
        $self->attributes[key] = value;
    }
    string getAttr(string key) {
        return $self->attributes[key];
    }
};

%newobject make_LinearLayer;
%newobject make_Logreglayer;
%newobject make_SoftmaxLayer;
%newobject make_TanhLayer;
%newobject make_ReluLayer;
%newobject make_Stacked;
%newobject make_Reversed;
%newobject make_Parallel;
%newobject make_MLP;
%newobject make_LSTM;
%newobject make_LSTM1;
%newobject make_BIDILSTM;

INetwork *make_LinearLayer();
INetwork *make_LogregLayer();
INetwork *make_SoftmaxLayer();
INetwork *make_TanhLayer();
INetwork *make_ReluLayer();
INetwork *make_Stacked();
INetwork *make_Reversed();
INetwork *make_Parallel();
INetwork *make_MLP();
INetwork *make_LSTM();
INetwork *make_LSTM1();
INetwork *make_BIDILSTM();

void forward_algorithm(Mat &lr,Mat &lmatch,double skip=-5.0);
void forwardbackward(Mat &both,Mat &lmatch);
void ctc_align_targets(Sequence &posteriors,Sequence &outputs,Sequence &targets);
void mktargets(Sequence &seq, Classes &targets, int ndim);

%inline %{
    Mat &getdebugmat() { return debugmat; }
    %}

%pythoncode %{
from numpy import *
class CNetwork:
    def __init__(self,net):
        self.net = net
    def init(self,*args):
        self.net.init(*args)
    def save(self,fname):
        self.net.save(fname)
    def load(self,fname):
        self.net.load(fname)
    def ninput(self):
        return self.net.input()
    def noutput(self):
        return self.net.noutput()
    def forward(self,xs):
        self.net.inputs.set(xs.astype(float32))
        self.net.forward()
        N = self.net.outputs.size()
        d = self.net.outputs[0].size()
        ys = zeros((N,d),'f')
        self.net.outputs.get(ys)
        return ys
    def backward(self,deltas):
        self.net.d_outputs.set(deltas.astype(float32))
        self.net.backward()
    def predict(self,xs):
        return self.forward()
    def train(self,xs,ys,debug=0):
        xs = array(xs)
        ys = array(ys)
        pred = self.forward(xs)
        deltas = ys - pred
        self.net.d_outputs.set(deltas)
        self.net.backward()
        return pred
    def ctrain(self,xs,cs,debug=0,lo=1e-5,accelerated=1):
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
        self.net.lr = r
        self.net.momentum = momentum
    def getState(self,name):
        seq = self.net.getState(name)
        a = zeros((seq.size(),seq[0].size()),'f')
        seq.get(a)
        return a
    def update(self):
        pass

def py_forward_algorithm(lmatch_):
    lmatch = Mat()
    lmatch.set(lmatch_.astype(float32))
    lr = Mat()
    forward_algorithm(lr,lmatch)
    result = zeros((lr.rows(),lr.cols()),'f')
    lr.get(result)
    return result

def py_forwardbackward(lmatch_):
    lmatch = Mat()
    lmatch.set(lmatch_.astype(float32))
    both = Mat()
    forwardbackward(both,lmatch)
    result = zeros((both.rows(),both.cols()),'f')
    both.get(result)
    return result

def py_ctc_align_targets(outputs_,targets_):
    outputs = Sequence()
    outputs.set(outputs_)
    targets = Sequence()
    targets.set(targets_)
    posteriors = Sequence()
    ctc_align_targets(posteriors,outputs,targets)
    result = zeros((posteriors.size(),posteriors[0].size()),'f')
    posteriors.get(result)
    return result
%}
