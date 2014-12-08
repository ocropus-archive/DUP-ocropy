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
    Sequence outputs,targets;
    Sequence aligned;
    outputs.resize(100);
    for(int t=0;t<outputs.size();t++) {
        outputs[t] = Vec::Random(5)*0.01;
        outputs[t]((5*t)/outputs.size()) = 1.0;
    }
    targets.resize(10);
    for(int t=0;t<targets.size();t++) {
        targets[t] = Vec::Random(5)*0.01;
        targets[t]((5*t)/targets.size()) = 1.0;
    }
    ctc_align_targets(aligned,outputs,targets);
    return 0;
}

