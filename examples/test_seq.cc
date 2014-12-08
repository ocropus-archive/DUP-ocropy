#include <math.h>
#include <memory>
#include <iostream>
#include <string>
#include "pyeigen.h"
#include "clstm.h"
#include "testsequences.h"

using namespace ocropus;
using namespace std;
using namespace pyeigen;

int main(int argc,char **argv) {
    shared_ptr<PyServer> python;
    NmodSequence test;
    shared_ptr<INetwork> net;
    net.reset(make_LSTM1());
    net->init(test.no,2,test.ni);
    net->lr = 1e-4;
    net->info("");
    test.cycle = 3;

    cout.precision(6);
    python.reset(make_PyServer());
    python->open();
    for(int trial=0;trial<1e9;trial++) {
        test.makeTest();
        net->ctrain(test.input,test.classes);
        if(trial%10000==0) {
            cout << "===" << trial << endl;
            python->clf();
            python->subplot(1,2,1);
            python->plot(timeslice(net->inputs,0).cast<float>(),"'y',alpha=0.5");
            python->plot(timeslice(net->outputs,1).cast<float>(), "'g'");
            python->plot(timeslice(test.target,1).cast<float>(),
                         "'b',linewidth=5,alpha=0.3");
            python->subplot(1,2,2);
            Sequence *states = net->getState(".lstm1.lstm.state");
            assert(states!=nullptr);
            python->plot2(timeslice(*states,0).cast<float>(),
                          timeslice(*states,1).cast<float>());
        }
    }
}
