#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <sys/time.h>

#include "multidim.h"
#include "h5multi.h"
#include "pymulti.h"

using namespace std;
using namespace Eigen;
using namespace ocropus;
using namespace h5multi;
using namespace pymulti;

namespace {
double now() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

template <class T>
inline void print(const T &arg) {
    cout << arg << endl;
}

template <class T, typename ... Args>
inline void print(T arg, Args ... args) {
    cout << arg << " ";
    print(args ...);
}

inline const char *getenv(const char *name, const char *dflt) {
    const char *result = std::getenv(name);
    if (result) return result;
    return dflt;
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

struct OcrDataset {
    string iname = "images";
    string oname = "transcripts";
    HDF5 h5;
    mdarray<int> codec;
    int nsamples = -1;
    int ndims = -1;
    int nclasses = -1;

    OcrDataset(const char *h5file) {
        H5::Exception::dontPrint();
        h5.open(h5file);
        mdarray<int> idims, odims;
        h5.shape(idims, iname.c_str());
        h5.shape(odims, oname.c_str());
        assert(idims(0) == odims(0));
        nsamples = idims(0);
        print("got", nsamples, "training samples");
        h5.get(codec, "codec");
        print("got codec:", codec.size());
        nclasses = codec.size();
        mdarray<float> a;
        h5.getdrow(a, 0, iname.c_str());
        assert(a.rank() == 2);
        ndims = a.dim(1);
        h5.getarow(a, 0, oname.c_str());
        assert(a.rank() == 1);
    }
    void image(mdarray<float> &a, int index) {
        h5.getdrow(a, index, iname.c_str());
        assert(a.rank() == 2);
        assert(a.dim(1) == ndims);
    }
    void transcript(mdarray<int> &a, int index) {
        h5.getarow(a, index, oname.c_str());
        assert(a.rank() == 1);
    }
    string to_string(mdarray<int> &transcript) {
        string result;
        for (int i = 0; i < transcript.size(); i++) {
            int label = transcript(i);
            int codepoint = codec(label);
            char chr = char(min(255, codepoint));
            result.push_back(chr);
        }
        return result;
    }
    string to_string(vector<int> &transcript) {
        mdarray<int> transcript_(int(transcript.size()));
        for (int i = 0; i < transcript.size(); i++) transcript_[i] = transcript[i];
        return to_string(transcript_);
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

bool anynan(mdarray<float> &a) {
    for (int i = 0; i < a.size(); i++)
        if (isnan(a[i])) return true;
    return false;
}

int main_ocr(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew.h5";
    string load_name = getsenv("load", "");
    double lrate = getdenv("lrate", 1e-4);
    string save_name = getsenv("save_name", "model-%08d.h5");
    int start = getienv("start", 0);
    int ntrain = getienv("ntrain", 1000000);
    int nhidden = getienv("hidden", 100);
    int save_every = getienv("save_every", 0);
    int display_every = getienv("display_every", 0);

    unique_ptr<PyServer> py;
    if (display_every > 0) {
        py.reset(new PyServer());
        if (display_every > 0) py->open();
        py->eval("ion()");
        py->eval("matplotlib.rc('xtick',labelsize=7)");
        py->eval("matplotlib.rc('ytick',labelsize=7)");
        py->eval("matplotlib.rcParams.update({'font.size':7})");
    }

    OcrDataset dataset(h5file);
    assign(codec, dataset.codec);

    shared_ptr<INetwork> net(make_BIDILSTM());
    net->init(dataset.nclasses, nhidden, dataset.ndims);
    net->setLearningRate(lrate, 0.9);
    assign(net->codec, dataset.codec);
    if (load_name != "") net->load(load_name.c_str());

    mdarray<float> image, outputs, aligned;
    mdarray<int> transcript;
    Sequence targets;
    Sequence saligned;
    Classes classes;

    double start_time = now();

    for (int trial = start; trial < ntrain; trial++) {
        int sample = trial % dataset.nsamples;
        if (sample > 0 && save_every > 0 && sample%save_every == 0) {
            char fname[4096];
            sprintf(fname, save_name.c_str(), sample);
            print("saving", fname);
            net->save(fname);
        }
        dataset.image(image, sample);
        float m = amax(image);
        for (int i = 0; i < image.size(); i++) image[i] /= m;
        dataset.transcript(transcript, sample);
        print(trial, sample,
              "dim", image.dim(0), image.dim(1),
              "time", now()-start_time,
              "lrate", lrate, "hidden", nhidden);
        print("TRU:", "'"+dataset.to_string(transcript)+"'");
        assign(net->inputs, image);
        net->forward();
        assign(classes, transcript);
        assign(outputs, net->outputs);
        mktargets(targets, classes, dataset.nclasses);
        ctc_align_targets(saligned, net->outputs, targets);
        assert(saligned.size() == net->outputs.size());
        net->d_outputs.resize(net->outputs.size());
        for (int t = 0; t < saligned.size(); t++)
            net->d_outputs[t] = saligned[t] - net->outputs[t];
        net->backward();
        assign(aligned, saligned);
        if (anynan(outputs) || anynan(aligned)) {
            print("got nan");
            break;
        }
        Classes output_classes, aligned_classes;
        trivial_decode(output_classes, net->outputs);
        trivial_decode(aligned_classes, saligned);
        string gt = dataset.to_string(transcript);;
        string out = dataset.to_string(output_classes);
        string aln = dataset.to_string(aligned_classes);
        print("OUT:", "'"+out+"'");
        print("ALN:", "'"+aln+"'");
        print(levenshtein(gt,out));

        if (display_every > 0 && sample%display_every == 0) {
            net->d_outputs.resize(saligned.size());
            py->eval("clf()");
            py->subplot(4, 1, 1);
            py->evalf("title('%s')", gt.c_str());
            py->imshowT(image, "cmap=cm.gray,interpolation='bilinear'");
            py->subplot(4, 1, 2);
            py->evalf("title('%s')", out.c_str());
            py->imshowT(outputs, "cmap=cm.hot,interpolation='bilinear'");
            py->subplot(4, 1, 3);
            py->evalf("title('%s')", aln.c_str());
            py->imshowT(aligned, "cmap=cm.hot,interpolation='bilinear'");
            py->subplot(4, 1, 4);
            mdarray<float> v;
            v.resize(outputs.dim(0));
            for (int t = 0; t < outputs.dim(0); t++)
                v(t) = outputs(t, 0);
            py->plot(v, "color='b'");
            int sp = 1;
            for (int t = 0; t < outputs.dim(0); t++)
                v(t) = outputs(t, sp);
            py->plot(v, "color='g'");
            int nclass = net->outputs[0].size();
            for (int t = 0; t < outputs.dim(0); t++)
                v(t) = net->outputs[t].segment(2, nclass-2).maxCoeff();
            py->evalf("xlim(0,%d)", outputs.dim(0));
            py->plot(v, "color='r'");
            py->eval("ginput(1,1e-3)");
        }
    }
}

int main_eval(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew-test.h5";
    string load_name = getsenv("load", "");
    OcrDataset dataset(h5file);
    shared_ptr<INetwork> net(make_BIDILSTM());
    int nhidden = 17;
    net->init(dataset.nclasses, nhidden, dataset.ndims);
    if(load_name=="") throw "must give load=";
    net->load(load_name.c_str());

    mdarray<float> image;
    mdarray<int> transcript;
    Classes classes;

    double total = 0;
    double errs = 0;

    for (int sample = 0; sample < dataset.nsamples; sample++) {
        dataset.image(image, sample);
        float m = amax(image);
        for (int i = 0; i < image.size(); i++) image[i] /= m;
        dataset.transcript(transcript, sample);
        assign(net->inputs, image);
        net->forward();
        Classes output_classes;
        trivial_decode(output_classes, net->outputs);
        string gt = dataset.to_string(transcript);;
        string out = dataset.to_string(output_classes);
        print(to_string(sample)+"\t"+out);
        total += gt.size();
        double err = levenshtein(gt,out);
        errs += err;
        cout.flush();
    }
    print("errs",errs,"total",total,"rate",errs*100.0/total);
    cout.flush();
}

int main_dump(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew-test.h5";
    OcrDataset dataset(h5file);
    for (int sample = 0; sample < dataset.nsamples; sample++) {
        mdarray<int> transcript;
        dataset.transcript(transcript, sample);
        string gt = dataset.to_string(transcript);;
        print(to_string(sample)+"\t"+gt);
        cout.flush();
    }
}

const char *usage = /*program+*/ R"(data.h5

data.h5 is an HDF5 file containing:

float images(N,*): text line images (or sequences of vectors)
int images_dims(N,2): shape of the images
int transcripts(N,*): corresponding transcripts
)";

int main(int argc, char **argv) {
    if (argc < 2) {
        print(string(argv[0])+" "+usage);
        exit(1);
    }
    try {
        if(getienv("dump")) {
            return main_dump(argc, argv);
        } else if(getienv("eval")) {
            return main_eval(argc, argv);
        } else {
            return main_ocr(argc, argv);
        }
    } catch(const char *msg) {
        print("EXCEPTION", msg);
    } catch(...) {
        print("UNKNOWN EXCEPTION");
    }
}

