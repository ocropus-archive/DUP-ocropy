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

int main(int argc,char **argv) {
    shared_ptr<INetwork> net(make_BIDILSTM());
    net->init(2,2,2);
    net->load("temp-00010000-lstm.h5");
    cout << "loaded..." << endl;
    net->networks("",[](string s,INetwork *net) {
        cerr << s << endl;
    });
    Sequence xs;
    xs.resize(17);
    for(int t=0;t<xs.size();t++) xs[t] = Vec::Zero(48);
    net->inputs = xs;
    net->forward();
    return 0;
}

