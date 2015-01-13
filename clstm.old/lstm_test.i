// -*- C++ -*-
// Test cases to be included from lstm.cc
// Activate with -DTEST -Dtest_foo=main during compilation.

#include "pyeigen.h"

using namespace ocropus;
using namespace pyeigen;

namespace {

struct Weights {
    vector<string> names;
    vector<Ref<Mat>> weights;
    vector<Ref<Mat>> deltas;
    Weights(Network &net,string prefix="net") {
        net.weights(prefix,[this](string s,Ref<Mat> w,Ref<Mat> d) {
            names.push_back(s);
            weights.push_back(w);
            deltas.push_back(d);
        });
    }
    void report() {
        for(int i=0;i<weights.size();i++) {
            Ref<Mat> &w = weights[i];
            Ref<Mat> &d = deltas[i];
            print(i,names[i],
                  "w",w.rows(),w.cols(),w.minCoeff(),w.maxCoeff(),
                  "d",d.minCoeff(),d.maxCoeff());
        }
    }
};

void random(Sequence &xs,int len,int d) {
    xs.resize(len);
    for(int t=0;t<len;t++)
        xs[t] = Vec::Random(d);
}

}

// Directly test the gradients.

int test_grad(int argc,char **argv) {
    for(int round=0;round<100;round++) {
        MLP net;
        int no = 1+rand()%10;
        int nh = 1+rand()%10;
        int ni = 1+rand()%10;
        int len = 1+rand()%10;
        net.init(no,nh,ni);
        Weights weights(net);
        if(round==0) weights.report();
        int w = rand() % weights.weights.size();
        int i = rand() % weights.weights[w].rows();
        int j = rand() % weights.weights[w].cols();
        Sequence xs,ys;
        random(xs,len,ni);
        random(ys,len,no);
        double eps = 0.01;
        net.lr = 1.0;
        no_update = true;
        double before = net.error2(xs,ys);
        double bpgrad = weights.deltas[w](i,j);
        weights.weights[w](i,j) += eps;
        double after = net.error2(xs,ys);
        double grad = (after-before)/eps;
        print(bpgrad/grad/-2,"@",w,i,j);
    }
    return 0;
}

// Verify forward propagation only.
// Verified numerical agreement with lstm.py 11/14

#define D(X) print1(X,#X)
int test_lstm(int argc,char **argv) {
    cout.precision(6);
    Sequence inputs,outputs,targets;
    inputs.resize(5);
    for(int t=0;t<inputs.size();t++) {
        inputs[t] = Vec::Zero(2);
        inputs[t](0) = (t==0);
    }
    LSTM net;
    net.init(2,2);
    net.lr = 1e-3;
    net.inputs = inputs;
    D(net.WCI);
    net.WCI << 1,2,3,4,5,6,7,8,9,10;
    net.WCI *= 0.1;
    net.WGI << 1,2,1,2,1,2,3,2,3,2;
    net.WGF << 1,0,-1,0,0,0,0,0,-1,0;
    net.WGO << 1,0,-1,3,0,0,0,0,-1,0;
    net.WIP << -1,1;
    net.WFP << 0.1,0.3;
    net.WOP << 0.9,0.2;
    // D(net.WCI); D(net.WGI); D(net.WGF); D(net.WGO);
    // D(net.WIP); D(net.WFP); D(net.WOP);
    D(net.inputs);
    net.forward();
    D(net.cix);
    D(net.ci);
    D(net.state);
    D(net.gox);
    D(net.go);
    D(net.outputs);
}

// Verify backward propagation and updates.
// Verified numerical agreement with lstm.py 11/14

int test_lstm2(int argc,char **argv) {
    cout.precision(6);
    Sequence inputs,outputs,targets,deltas;
    inputs.resize(5);
    deltas.resize(5);
    for(int t=0;t<inputs.size();t++) {
        inputs[t] = Vec::Zero(2);
        inputs[t](0) = 1;
        deltas[t] = Vec::Zero(2);
        deltas[t](0) = t;
    }
    LSTM net;
    net.init(2,2);
    net.lr = 1e-3;
    net.momentum = 1.0;         // needed to preserve D...
    net.inputs = inputs;
    net.WCI << 1,2,3,4,5,6,7,8,9,10;
    net.WCI *= 0.1;
    net.WGI << 1,1,2,2,2,0,0,0,2,0;
    net.WGF << 0,0,1,0,0,0,0,2,0,0;
    net.WGO << 0,0,0,1,0,0,2,0,0,0;
    net.WIP << -1,1;
    net.WFP << 2,-1;
    net.WOP << -1,2;
    net.forward();
    D(net.outputs);
    net.d_outputs = deltas;
    net.backward();
    D(net.DWCI);
    D(net.DWGI);
    D(net.DWGF);
    D(net.DWGO);
    D(net.DWIP);
    D(net.DWFP);
    D(net.DWOP);
}

// Verify combination LSTM+Softmax
// Verified numerical agreement with lstm.py 11/14

int test_lstm3(int argc,char **argv) {
    cout.precision(6);
    Sequence inputs,outputs,targets,deltas;
    inputs.resize(5);
    deltas.resize(5);
    for(int t=0;t<inputs.size();t++) {
        inputs[t] = Vec::Zero(2);
        inputs[t](0) = t*0.1;
        deltas[t] = Vec::Zero(2);
        deltas[t](0) = t;
    }
    LSTM1 net;
    net.init(2,2,2);
    net.lstm.lr = 1e-3;
    net.momentum = 1.0;// needed to preserve D...
    net.inputs = inputs;

    net.lstm.WCI << 1,2,3,4,5,6,7,8,9,10;
    net.lstm.WCI *= 0.1;
    net.lstm.WGI << 1,1,2,2,2,0,0,0,2,0;
    net.lstm.WGF << 0,0,1,0,0,0,0,2,0,0;
    net.lstm.WGO << 0,0,0,1,0,0,2,0,0,0;
    net.lstm.WIP << -1,1;
    net.lstm.WFP << 2,-1;
    net.lstm.WOP << -1,2;
    net.logreg.W << 2,3,-5,-6;
    net.logreg.W *= 0.1;
    net.logreg.w << 1,-4;
    net.logreg.w *= 0.1;
    net.forward();

    D(net.lstm.outputs);
    D(net.logreg.outputs);

    net.d_outputs = deltas;
    net.backward();

    D(net.lstm.d_outputs);
    D(net.lstm.d_inputs);

    D(net.lstm.cierr);
    D(net.lstm.source);
    D(net.lstm.DWCI);
    D(net.lstm.DWGI);
    D(net.lstm.DWGF);
    D(net.lstm.DWGO);
    D(net.lstm.DWIP);
    D(net.lstm.DWFP);
    D(net.lstm.DWOP);
}

// Verify combination LSTM+Softmax with ctrain
// Verified numerical agreement with lstm.py 11/14

int test_lstm4(int argc,char **argv) {
    cout.precision(6);
    Sequence inputs,outputs,targets,deltas;
    Classes classes;
    inputs.resize(5);
    classes.resize(5);
    for(int t=0;t<inputs.size();t++) {
        inputs[t] = Vec::Zero(2);
        inputs[t](0) = t*0.1;
        classes[t] = t%2;
    }
    LSTM1 net;
    net.init(2,2,2);
    net.lstm.lr = 1e-3;
    net.momentum = 1.0;// needed to preserve D...
    net.inputs = inputs;

    net.lstm.WCI << 1,2,3,4,5,6,7,8,9,10;
    net.lstm.WCI *= 0.1;
    net.lstm.WGI << 1,1,2,2,2,0,0,0,2,0;
    net.lstm.WGF << 0,0,1,0,0,0,0,2,0,0;
    net.lstm.WGO << 0,0,0,1,0,0,2,0,0,0;
    net.lstm.WIP << -1,1;
    net.lstm.WFP << 2,-1;
    net.lstm.WOP << -1,2;
    net.logreg.W << 2,3,-5,-6;
    net.logreg.W *= 0.1;
    net.logreg.w << 1,-4;
    net.logreg.w *= 0.1;
    net.ctrain_accelerated(inputs,classes);

    D(net.lstm.outputs);
    D(net.logreg.outputs);
    D(net.lstm.d_outputs);
    D(net.lstm.d_inputs);
    D(net.lstm.cierr);
    D(net.lstm.source);
    D(net.lstm.DWCI);
    D(net.lstm.DWGI);
    D(net.lstm.DWGF);
    D(net.lstm.DWGO);
    D(net.lstm.DWIP);
    D(net.lstm.DWFP);
    D(net.lstm.DWOP);
}

