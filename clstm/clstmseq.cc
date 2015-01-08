#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>

#include "multidim.h"
#include "h5multi.h"
#include "pymulti.h"

using namespace std;
using namespace Eigen;
using namespace ocropus;
using namespace h5multi;
using namespace pymulti;

namespace {
template <class T>
inline void print(const T &arg) {
    cout << arg << endl;
}

template <class T, typename ... Args>
inline void print(T arg, Args ... args) {
    cout << arg << " ";
    print(args ...);
}

template <class S, class T>
void assign(S &dest, T &src) {
    dest.resize_(src.dims);
    int n = dest.size();
    for (int i = 0; i < n; i++) dest.data[i] = src.data[i];
}

template <class S, class T>
void transpose(S &dest, T &src) {
    dest.resize(src.dim(1), src.dim(0));
    for (int i = 0; i < dest.dim(0); i++)
        for (int j = 0; j < dest.dim(1); j++)
            dest(i, j) = src(j, i);
}

template <class T>
void transpose(T &a) {
    T temp;
    transpose(temp, a);
    assign(a, temp);
}

template <class T>
void assign(Sequence &seq, T &a) {
    assert(a.rank() == 2);
    seq.resize(a.dim(0));
    for (int t = 0; t < a.dim(0); t++) {
        seq[t].resize(a.dim(1));
        for (int i = 0; i < a.dim(1); i++) seq[t](i) = a(t, i);
    }
}

template <class T>
void assign(T &a, Sequence &seq) {
    a.resize(int(seq.size()), int(seq[0].size()));
    for (int t = 0; t < a.dim(0); t++) {
        for (int i = 0; i < a.dim(1); i++) a(t, i) = seq[t](i);
    }
}

void assign(Classes &classes, mdarray<int> &transcript) {
    classes.resize(transcript.size());
    for (int i = 0; i < transcript.size(); i++)
        classes[i] = transcript(i);
}

struct SeqDataset {
    string iname = "inputs";
    string oname = "outputs";
    HDF5 h5;
    int nsamples = -1;
    int nin = -1;
    int nout = -1;

    SeqDataset(const string &h5file) {
        H5::Exception::dontPrint();
        h5.open(h5file);
        mdarray<int> idims, odims;
        h5.shape(idims, iname);
        h5.shape(odims, oname);
        assert(idims(0) == odims(0));
        nsamples = idims(0);
        print("got", nsamples, "training samples");
        mdarray<float> a;
        h5.getdrow(a, 0, iname);
        assert(a.rank() == 2);
        nin = a.dim(1);
        h5.getdrow(a, 0, oname);
        assert(a.rank() == 2);
        nout = a.dim(1);
        if (nout == 1) nout = 2;
    }
    void input(mdarray<float> &a, int index) {
        h5.getdrow(a, index, iname);
        assert(a.rank() == 2);
        assert(a.dim(1) == nin);
    }
    void output(mdarray<float> &a, int index) {
        h5.getdrow(a, index, oname);
        if (a.dim(1) == 1) {
            mdarray<float> temp;
            assign(temp, a);
            a.resize(temp.dim(0), 2);
            for (int t = 0; t < a.dim(0); t++) {
                a(t, 0) = 1-temp(t, 0);
                a(t, 1) = temp(t, 0);
            }
        }
        assert(a.rank() == 2);
        assert(a.dim(1) == nout);
    }
};

template <class A, class T>
int indexof(A &a, const T &t) {
    for (int i = 0; i < a.size(); i++)
        if (a[i] == t) return i;
    return -1;
}

template <class T>
T amin(mdarray<T> &a) {
    T m = a[0];
    for (int i = 1; i < a.size(); i++) if (a[i] < m) m = a[i];
    return m;
}

template <class T>
T amax(mdarray<T> &a) {
    T m = a[0];
    for (int i = 1; i < a.size(); i++) if (a[i] > m) m = a[i];
    return m;
}
}

mdarray<int> codec;

void debug_decode(Sequence &outputs, Sequence &aligned) {
    for (int t = 0; t < outputs.size(); t++) {
        int oindex, aindex;
        outputs[t].maxCoeff(&oindex);
        aligned[t].maxCoeff(&aindex);
        print(t,
              "outputs", outputs[t](0), outputs[t](1),
              oindex, outputs[t](oindex),
              "aligned", aligned[t](0), aligned[t](1),
              aindex, aligned[t](aindex));
    }
}

void trivial_decode(Classes &cs, Sequence &outputs) {
    int N = outputs.size();
    int t = 0;
    float mv = 0;
    int mc = -1;
    while (t < N) {
        int index;
        float v = outputs[t].maxCoeff(&index);
        if (index == 0) {
            // NB: there should be a 0 at the end anyway
            if (mc != -1) cs.push_back(mc);
            mv = 0; mc = -1; t++;
            continue;
        }
        if (v > mv) {
            mv = v;
            mc = index;
        }
        t++;
    }
}

void getslice(mdarray<float> &a, Sequence &seq, int i) {
    a.resize(int(seq.size()));
    for (int t = 0; t < seq.size(); t++) a(t) = seq[t][i];
}

int main_seq(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew.h5";
    float lr = 1e-5;

    PyServer py;
    py.open();

    SeqDataset dataset(h5file);
    shared_ptr<INetwork> net;

    string mode = getsenv("mode", "lstm");
    if (mode == "bidi") net.reset(make_BIDILSTM());
    else if (mode == "revlstm") net.reset(make_REVLSTM1());
    else if (mode == "lstm") net.reset(make_LSTM1());
    else throw "unknown mode";
    int nstates = getienv("states", 2);

    net->init(dataset.nout, nstates, dataset.nin);
    double lrate = getdenv("lrate", 1e-4);
    net->setLearningRate(lrate, 0.9);

    if (getienv("debug_nets")) {
        net->networks("net", [] (string name, INetwork *net) {
                          print("net", name, net->lr, net->momentum);
                      });
    }
    if (getienv("debug_states")) {
        net->states("net", [] (string name, Sequence *seq) {
                        print("state", name, seq->size());
                    });
    }

    mdarray<float> image, outputs, aligned;
    mdarray<int> transcript;
    Sequence targets;
    Sequence saligned;
    Classes classes;
    for (int sample = 0; sample < dataset.nsamples; sample++) {
        mdarray<float> input, target;
        dataset.input(input, sample);
        dataset.output(target, sample);
        Sequence input_;
        assign(input_, input);
        net->setInputs(input_);
        net->forward();
        Sequence target_;
        assign(target_, target);
        net->setTargets(target_);
        net->backward();
        if (sample%1000 == 0) {
            py.eval("clf()");
            py.eval("subplot(121)");
            py.eval("ylim(-.1,1.1)");
            mdarray<float> a;
            getslice(a, input_, 0);
            py.plot(a, "color='y',linewidth=5");
            getslice(a, target_, 1);
            py.plot(a, "color='b',linewidth=2");
            getslice(a, net->outputs, 1);
            print("outputs", amin(a), amax(a));
            py.plot(a, "color='r',ls='--'");
            Sequence *state = 0;
            state = net->getState(".lstm1.lstm.state");
            if (state) {
                mdarray<float> b;
                getslice(a, *state, 0);
                getslice(b, *state, 1);
                py.eval("subplot(122)");
                py.plot2(a, b);
            }
            py.eval("ginput(1,0.005)");
        }
    }
}

const char *usage =
    /*program+*/ R"(data.h5

data.h5 is an HDF5 file containing:

float inputs(N,*): input sequences
int inputs_dims(N,2): shape of input sequences
float outputs(N,*): output sequences
int outputs_dims(N,2): shape of output sequences
)";

int main(int argc, char **argv) {
    if (argc < 2) {
        print(string(argv[0])+" "+usage);
        exit(1);
    }
    try {
        return main_seq(argc, argv);
    } catch(const char *msg) {
        print("EXCEPTION", msg);
    } catch(...) {
        print("UNKNOWN EXCEPTION");
    }
}

