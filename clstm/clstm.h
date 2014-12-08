// -*- C++ -*-

#ifndef ocropus_lstm__
#define ocropus_lstm__

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <memory>
#include <map>
#include <Eigen/Dense>

namespace ocropus {

using namespace std;
using Eigen::Ref;

#ifdef LSTM_DOUBLE
typedef double Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
#else
typedef float Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXf Vec;
typedef Eigen::MatrixXf Mat;
#endif

typedef vector<Vec> Sequence;
typedef vector<int> Classes;

inline Vec timeslice(const Sequence &s,int i) {
    Vec result(s.size());
    for(int t=0;t<s.size();t++)
        result[t] = s[t][i];
    return result;
}

struct VecMat {
    Vec *vec = 0;
    Mat *mat = 0;
    VecMat() {}
    VecMat(Vec *vec) { this->vec = vec; }
    VecMat(Mat *mat) { this->mat = mat; }
};

struct INetwork {
    virtual ~INetwork() {}
    string name = "???";
    Float softmax_floor = 1e-5;
    bool softmax_accel = false;
    Float lr = 1e-4;
    Float momentum = 0.9;
    Sequence inputs,d_inputs;
    Sequence outputs,d_outputs;
    vector<shared_ptr<INetwork>> sub;
    map<string,string> attributes;
    virtual void init(int no,int ni) { throw "unimplemented"; }
    virtual void init(int no,int nh,int ni) { throw "unimplemented"; }
    virtual void init(int no,int nh2,int nh,int ni) { throw "unimplemented"; }
    virtual void add(shared_ptr<INetwork> net) { sub.push_back(net); }
    virtual void forward() = 0;
    virtual void backward() = 0;
    typedef function<void (const string &,VecMat,VecMat)> WeightFun;
    typedef function<void (const string &,Sequence *)> StateFun;
    virtual void myweights(const string &prefix,WeightFun f) {}
    virtual void mystates(const string &prefix,StateFun f) {}
    virtual int ninput() { return -999999; }
    virtual int noutput() { return -999999; }
    virtual void preSave() {}
    virtual void postLoad() {}

    // move the rest out of this class
    void train(Sequence &xs,Sequence &targets);
    void ctrain(Sequence &xs,Classes &cs);
    void ctrain_accelerated(Sequence &xs,Classes &cs,Float lo=1e-5);
    void cpred(Classes &preds,Sequence &xs);
    void info(string prefix);
    void weights(const string &prefix,WeightFun f);
    void states(const string &prefix,StateFun f);
    void networks(const string &prefix,function<void (string,INetwork*)>);
    Sequence *getState(string name);
    void save(const char *fname);
    void load(const char *fname);
};

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

extern Mat debugmat;

}

#endif
