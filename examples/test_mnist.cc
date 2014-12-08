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

    int bs = 10;
    shared_ptr<INetwork> net;
    net = shared_ptr<INetwork>(make_Stacked());
    INetwork *l1 = make_LogregLayer();
    l1->init(200,784);
    net->add(shared_ptr<INetwork>(l1));
    INetwork *l2 = make_LogregLayer();
    l2->init(10,200);
    net->add(shared_ptr<INetwork>(l2));

    net->lr = 0.5/bs;
    net->momentum = 0;

    print("lr",net->lr,net->lr*bs,"bs",bs,"momentum",net->momentum);

    for(int epoch=0;epoch<100;epoch++) {
        for(int start=0;start<60000;start+=bs) {
            Sequence xs;
            Classes cs;
            xs.resize(bs);
            cs.resize(bs);
            for(int t=0;t<bs;t++) {
                xs[t] = deskewed.row(start+t);
                cs[t] = labels(start+t);
            }
            net->ctrain(xs,cs);
        }
        // weights.report();
        int errs = 0;
        for(int t=0;t<10000;t++) {
            Sequence xs;
            Classes preds;
            xs.resize(1);
            xs[0] = test_deskewed.row(t);
            net->cpred(preds,xs);
            errs += (preds[0] != test_labels(t));
            // if(t<10) print(t,preds[0],test_labels(t,0));
        }
        print("epoch",epoch,errs);
    }
    return 0;
}

