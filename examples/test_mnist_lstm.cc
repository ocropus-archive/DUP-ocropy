// Test MLP implementation against MNIST.

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <assert.h>
#include "h5eigen.h"
#include "pyeigen.h"
#include "clstm.h"

using namespace ocropus;
using namespace h5eigen;

template <typename T>
void print(T arg) {
    cout << arg << endl;
}
template <typename T,typename... Args>
void print(T arg,Args&&... args) {
    cout << arg << " ";
    print(args...);
}

template <class T>
inline void transpose(T &m) {
    T temp = m.transpose();
    m = temp;
}

void seqOfImage(Sequence &s,Vec &x,int rows,int cols) {
    assert(x.size()==rows*cols);
    s.resize(cols);
    for(int t=0;t<cols;t++) {
        s[t] = Vec::Zero(rows);
        for(int i=0;i<rows;i++) {
            s[t][i] = x[i*cols+t];
        }
    }
}

int main(int argc,char **argv) {
    Mat deskewed,test_deskewed;
    iVec labels,test_labels;

    unique_ptr<HDF5> h5(make_HDF5());
    h5->open("mnist.h5");
    h5->get(deskewed,"deskewed"); transpose(deskewed);
    h5->get(labels,"labels");
    h5->get(test_deskewed,"test_deskewed"); transpose(test_deskewed);
    h5->get(test_labels,"test_labels");
    h5.reset();

    print("deskewed",deskewed.rows(),deskewed.cols());
    print("labels",labels.rows(),labels.cols());

    shared_ptr<INetwork> net;
    net.reset(make_BIDILSTM());
    net->init(11,100,28);
    net->lr = 1e-2;
    net->momentum = 0;

    print("lr",net->lr,"momentum",net->momentum);

    for(int epoch=0;epoch<100;epoch++) {
        for(int sample=0;sample<60000;sample++) {
            Vec x = deskewed.row(sample);
            Sequence xs;
            seqOfImage(xs,x,28,28);
            Classes cs;
            cs.resize(28);
            for(int t=0;t<28;t++) cs[t] = 10;
            cs[14] = labels[sample];
            net->ctrain(xs,cs);
        }
        net->save("mnist.lstm");
        // weights.report();
        int errs = 0;
        for(int sample=0;sample<10000;sample++) {
            Vec x = test_deskewed.row(sample);
            Sequence xs;
            seqOfImage(xs,x,28,28);
            Classes preds;
            net->cpred(preds,xs);
            errs += (preds[14] != test_labels[sample]);
        }
        print("epoch",epoch,errs);
    }
    return 0;
}

