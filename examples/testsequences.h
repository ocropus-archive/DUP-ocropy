// Test sequence generators.

#include "lstm.h"

namespace ocropus {

struct TestSequence {
    int N = 20;
    int ni = 2;
    int no = 2;
    Sequence input;
    Sequence output;
    Sequence target;
    Classes classes;

    // Generate a new sequence.

    void reset() {
        input.resize(N);
        output.resize(N);
        target.resize(N);
        classes.resize(N);
        for(int t=0;t<N;t++) {
            input[t] = Vec::Zero(ni);
            output[t] = Vec::Zero(no);
            target[t] = Vec::Zero(no);
            classes[t] = 0;
        }
    }

    // Internal method called after setting the classes
    // array to generate actual targets.
    
    void updateTargets() {
        int N = classes.size();
        for(int t=0;t<N;t++) {
            assert(unsigned(classes[t])<unsigned(target[t].size()));
            target[t](classes[t]) = 1.0;
        }
    }
};

// Output is the same as input.

struct IdentitySequence : TestSequence {
    int bs = 3;
    void makeTest() {
        reset();
        for(int t=0;t<N;t++) {
            int c = random()%ni;
            for(int i=0;i<no;i++) {
                input[t](i) = (i==c);
                classes[t] = c;
            }
        }
        updateTargets();
    }
};

// Output is a delayed version of a random input sequence.

struct DelaySequence : TestSequence {
    static inline int argmax(Vec &v) {
        int index = -1;
        v.maxCoeff(&index);
        return index;
    }
    int bs = 3;
    int delay = 1;
    void makeTest() {
        reset();
        for(int t=0;t<N;t++) {
            int c = random()%ni;
            for(int i=0;i<no;i++) {
                input[t](i) = (i==c);
                if(t-delay>=0) target[t](i) = argmax(target[t-delay]);
            }
        }
        updateTargets();
    }
};

// Generate an output every `cycle` steps, independent of input.
// Generates a little bit of `noise` and a downbeat at the start.

struct CyclicSequence : TestSequence {
    int bs = 1;
    int cycle = 3;
    int offset = 0;
    float noise = 0.01;
    float beat = 1;
    void makeTest() {
        reset();
        for(int t=0;t<N;t++) {
            for(int i=0;i<no;i++) {
                input[t](i) = noise*drand48() + beat*(t==0);
                classes[t] = (t%cycle==offset);
            }
        }
        updateTargets();
    }
};

// Generate an output for every `cycle` inputs.

struct NmodSequence : TestSequence {
    float p = 0.4;
    int cycle = 3;
    NmodSequence() { N = 40; }
    void makeTest() {
        reset();
        int count = 0;
        int last = 0;
        for(int t=0;t<N;t++) {
            int ibit = (!last && drand48()<p);
            last = ibit;
            int obit = (ibit && count%cycle==0);
            count += ibit;
            input[t](0) = ibit;
            classes[t] = obit;
        }
        updateTargets();
    }
};

}
