#include "clstm.h"
#include "h5eigen.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {
Mat debugmat;

using namespace std;
using Eigen::Ref;

bool no_update = false;
bool verbose = false;

void INetwork::setInputs(Sequence &inputs) {
    this->inputs.resize(inputs.size());
    for (int t = 0; t < this->inputs.size(); t++)
        this->inputs[t] = inputs[t];
}

void INetwork::setTargets(Sequence &targets) {
    assert(outputs.size() == targets.size());
    d_outputs.resize(outputs.size());
    for (int t = 0; t < outputs.size(); t++)
        d_outputs[t] = targets[t] - outputs[t];
}

void INetwork::setClasses(Classes &classes) {
    assert(outputs.size() == classes.size());
    d_outputs.resize(outputs.size());
    for (int t = 0; t < outputs.size(); t++) {
        d_outputs[t] = -outputs[t];
        d_outputs[t](classes[t]) += 1;
    }
}

void INetwork::train(Sequence &xs, Sequence &targets) {
    assert(xs.size() > 0);
    assert(xs.size() == targets.size());
    inputs = xs;
    forward();
    setTargets(targets);
    backward();
}

void INetwork::ctrain(Sequence &xs, Classes &cs) {
    inputs = xs;
    forward();
    int len = outputs.size();
    assert(len > 0);
    int N = outputs[0].size();
    assert(N > 0);
    d_outputs.resize(len);
    if (N == 1) {
        for (int t = 0; t < len; t++)
            d_outputs[t](0) = cs[t] ? 1.0-outputs[t](0) : -outputs[t](0);
    } else {
        for (int t = 0; t < len; t++) {
            d_outputs[t] = -outputs[t];
            int c = cs[t];
            d_outputs[t](c) = 1-outputs[t](c);
        }
    }
    backward();
}

void INetwork::ctrain_accelerated(Sequence &xs, Classes &cs, Float lo) {
    inputs = xs;
    forward();
    int len = outputs.size();
    assert(len > 0);
    int N = outputs[0].size();
    assert(N > 0);
    d_outputs.resize(len);
    if (N == 1) {
        for (int t = 0; t < len; t++) {
            if (cs[t] == 0)
                d_outputs[t](0) = -1.0/fmax(lo, 1.0-outputs[t](0));
            else
                d_outputs[t](0) = 1.0/fmax(lo, outputs[t](0));
        }
    } else {
        for (int t = 0; t < len; t++) {
            d_outputs[t] = -outputs[t];
            int c = cs[t];
            d_outputs[t](c) = 1.0/fmax(lo, outputs[t](c));
        }
    }
    backward();
}

void INetwork::cpred(Classes &preds, Sequence &xs) {
    inputs = xs;
    preds.resize(xs.size());
    forward();
    for (int t = 0; t < outputs.size(); t++) {
        int index = -1;
        outputs[t].maxCoeff(&index);
        preds[t] = index;
    }
}

void INetwork::info(string prefix){
    string nprefix = prefix + "." + name;
    cout << nprefix << ": " << lr << " " << momentum << " ";
    cout << inputs.size() << " " << (inputs.size() > 0 ? inputs[0].size() : -1) << " ";
    cout << outputs.size() << " " << (outputs.size() > 0 ? outputs[0].size() : -1) << endl;
    for (auto s : sub) s->info(nprefix);
}

void INetwork::weights(const string &prefix, WeightFun f) {
    string nprefix = prefix + "." + name;
    myweights(nprefix, f);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->weights(nprefix+to_string(i), f);
    }
}

void INetwork::states(const string &prefix, StateFun f) {
    string nprefix = prefix + "." + name;
    f(nprefix+".inputs", &inputs);
    f(nprefix+".d_inputs", &d_inputs);
    f(nprefix+".outputs", &outputs);
    f(nprefix+".d_outputs", &d_outputs);
    mystates(nprefix, f);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->states(nprefix+to_string(i), f);
    }
}

void INetwork::networks(const string &prefix, function<void (string, INetwork*)> f) {
    string nprefix = prefix+"."+name;
    f(nprefix, this);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->networks(nprefix, f);
    }
}

Sequence *INetwork::getState(string name) {
    Sequence *result = nullptr;
    states("", [&result, &name](const string &prefix, Sequence *s) {
               if (prefix == name) result = s;
           });
    return result;
}

void INetwork::save(const char *fname) {
    using namespace h5eigen;
    unique_ptr<HDF5> h5(make_HDF5());
    h5->open(fname, "w");
#ifdef USE_ATTRS
    h5->setAttr("ocropus", "0.0");
    for (auto &kv : attributes) {
    h5->setAttr(kv.first, kv.second);
    }
#endif
    weights("", [&h5](const string &prefix, VecMat a, VecMat da) {
                if (a.mat) h5->put(*a.mat, prefix.c_str());
                else if (a.vec) h5->put(*a.vec, prefix.c_str());
                else throw "oops (save type)";
            });
}

void INetwork::load(const char *fname) {
    using namespace h5eigen;
    unique_ptr<HDF5> h5(make_HDF5());
    h5->open(fname);
#ifdef USE_ATTRS
    h5->getAttrs(attributes);
#endif
    weights("", [&h5](const string &prefix, VecMat a, VecMat da) {
                if (a.mat) h5->get(*a.mat, prefix.c_str());
                else if (a.vec) h5->get1d(*a.vec, prefix.c_str());
                else throw "oops (load type)";
            });
    networks("", [] (string s, INetwork *net) {
                 net->postLoad();
             });
}

struct Network : INetwork {
    Float error2(Sequence &xs, Sequence &targets) {
        inputs = xs;
        forward();
        Float total = 0.0;
        d_outputs.resize(outputs.size());
        for (int t = 0; t < outputs.size(); t++) {
            Vec delta = targets[t] - outputs[t];
            total += delta.array().square().sum();
            d_outputs[t] = delta;
        }
        backward();
        return total;
    }
};

inline Float limexp(Float x) {
#if 1
    if (x < -MAXEXP) return exp(-MAXEXP);
    if (x > MAXEXP) return exp(MAXEXP);
    return exp(x);
#else
    return exp(x);
#endif
}

inline Float sigmoid(Float x) {
#if 1
    return 1.0 / (1.0 + limexp(-x));
#else
    return 1.0 / (1.0 + exp(-x));
#endif
}

template <class NONLIN>
struct Full : Network {
    Mat W;
    Vec w;
    Full() {
        name = "full";
    }
    int noutput() {
        return W.rows();
    }
    int ninput() {
        return W.cols();
    }
    void init(int no, int ni) {
        W = Mat::Random(no, ni) * 0.01;
        w = Vec::Random(no) * 0.01;
    }
    void forward() {
        outputs.resize(inputs.size());
        for (int t = 0; t < inputs.size(); t++) {
            outputs[t] = (W * inputs[t] + w);
            NONLIN::f(outputs[t]);
        }
    }
    void backward() {
        d_inputs.resize(d_outputs.size());
        for (int t = d_outputs.size()-1; t >= 0; t--) {
            NONLIN::df(d_outputs[t], outputs[t]);
            d_inputs[t] = W.transpose() * d_outputs[t];
        }
        if (no_update) return;
        for (int t = 0; t < d_outputs.size(); t++) {
            W += lr * d_outputs[t] * inputs[t].transpose();
            w += lr * d_outputs[t];
        }
        d_outputs[0](0, 0) = NAN;
    }
    void weights(const string &prefix, WeightFun f) {
        f(prefix+".W", &W, (Mat*)0);
        f(prefix+".w", &w, (Vec*)0);
    }
};

struct NoNonlin {
    template <class T>
    static void f(T &x) {
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
    }
};

struct SigmoidNonlin {
    template <class T>
    static void f(T &x) {
        x = x.unaryExpr(ptr_fun(sigmoid));
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= y.array() * (1-y.array());
    }
};

Float tanh_(Float x) {
    return tanh(x);
}
struct TanhNonlin {
    template <class T>
    static void f(T &x) {
        x = x.unaryExpr(ptr_fun(tanh_));
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= (1 - y.array().square());
    }
};

struct ReluNonlin {
    template <class T>
    static void f(T &x) {
        x = x.unaryExpr([] (Float x) { return fmax(0, x); });
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= (y.array() > 0);
    }
};

typedef Full<NoNonlin> LinearLayer;
typedef Full<SigmoidNonlin> LogregLayer;
typedef Full<TanhNonlin> TanhLayer;
typedef Full<ReluNonlin> ReluLayer;

INetwork *make_LinearLayer() {
    return new LinearLayer();
}

INetwork *make_LogregLayer() {
    return new LogregLayer();
}

INetwork *make_TanhLayer() {
    return new TanhLayer();
}

INetwork *make_ReluLayer() {
    return new TanhLayer();
}

struct SoftmaxLayer : Network {
    Mat W, d_W;
    Vec w, d_w;
    SoftmaxLayer() {
        name = "softmax";
    }
    int noutput() {
        return W.rows();
    }
    int ninput() {
        return W.cols();
    }
    void init(int no, int ni) {
        W = Mat::Random(no, ni) * 0.01;
        w = Vec::Random(no) * 0.01;
        d_W = Mat::Zero(no, ni);
        d_w = Vec::Zero(no);
    }
    void forward() {
        outputs.resize(inputs.size());
        for (int t = 0; t < inputs.size(); t++) {
            outputs[t] = (W * inputs[t] + w).array().unaryExpr(ptr_fun(limexp));
            Float total = fmax(outputs[t].sum(), 1e-9);
            outputs[t] /= total;
        }
    }
    void backward() {
        d_inputs.resize(d_outputs.size());
        for (int t = d_outputs.size()-1; t >= 0; t--) {
            d_inputs[t] = W.transpose() * d_outputs[t];
        }
        d_W *= momentum;
        d_w *= momentum;
        for (int t = 0; t < d_outputs.size(); t++) {
            d_W += lr * d_outputs[t] * inputs[t].transpose();
            d_w += lr * d_outputs[t];
        }
        if (no_update) return;
        W += d_W;
        w += d_w;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".W", &W, &d_W);
        f(prefix+".w", &w, &d_w);
    }
};

INetwork *make_SoftmaxLayer() {
    return new SoftmaxLayer();
}

struct Stacked : Network {
    Stacked() {
        name = "stacked";
    }
    int noutput() {
        return sub[sub.size()-1]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        assert(inputs.size() > 0);
        assert(sub.size() > 0);
        for (int n = 0; n < sub.size(); n++) {
            if (n == 0) sub[n]->inputs = inputs;
            else sub[n]->inputs = sub[n-1]->outputs;
            sub[n]->forward();
        }
        outputs = sub[sub.size()-1]->outputs;
        assert(outputs.size() == inputs.size());
    }
    void backward() {
        assert(outputs.size() > 0);
        assert(outputs.size() == inputs.size());
        assert(d_outputs.size() > 0);
        assert(d_outputs.size() == outputs.size());
        for (int n = sub.size()-1; n >= 0; n--) {
            if (n+1 == sub.size()) sub[n]->d_outputs = d_outputs;
            else sub[n]->d_outputs = sub[n+1]->d_inputs;
            sub[n]->backward();
        }
        d_inputs = sub[0]->d_inputs;
    }
};

INetwork *make_Stacked() {
    return new Stacked();
}

template <class T>
inline void revcopy(vector<T> &out, vector<T> &in) {
    int N = in.size();
    out.resize(N);
    for (int i = 0; i < N; i++) out[i] = in[N-i-1];
}

struct Reversed : Network {
    Reversed() {
        name = "reversed";
    }
    int noutput() {
        return sub[0]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        assert(sub.size() == 1);
        INetwork *net = sub[0].get();
        revcopy(net->inputs, inputs);
        net->forward();
        revcopy(outputs, net->outputs);
    }
    void backward() {
        assert(sub.size() == 1);
        INetwork *net = sub[0].get();
        assert(outputs.size() > 0);
        assert(outputs.size() == inputs.size());
        assert(d_outputs.size() > 0);
        revcopy(net->d_outputs, d_outputs);
        net->backward();
        revcopy(d_inputs, net->d_inputs);
    }
};

INetwork *make_Reversed() {
    return new Reversed();
}

struct Parallel : Network {
    Parallel() {
        name = "parallel";
    }
    int noutput() {
        return sub[0]->noutput() + sub[1]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        assert(sub.size() == 2);
        INetwork *net1 = sub[0].get();
        INetwork *net2 = sub[1].get();
        net1->inputs = inputs;
        net2->inputs = inputs;
        net1->forward();
        net2->forward();
        int N = inputs.size();
        assert(net1->outputs.size() == N);
        assert(net2->outputs.size() == N);
        int n1 = net1->outputs[0].size();
        int n2 = net2->outputs[0].size();
        outputs.resize(N);
        for (int t = 0; t < N; t++) {
            outputs[t].resize(n1+n2);
            outputs[t].segment(0, n1) = net1->outputs[t];
            outputs[t].segment(n1, n2) = net2->outputs[t];
        }
    }
    void backward() {
        assert(sub.size() == 2);
        INetwork *net1 = sub[0].get();
        INetwork *net2 = sub[1].get();
        assert(outputs.size() > 0);
        assert(outputs.size() == inputs.size());
        assert(d_outputs.size() > 0);
        int n1 = net1->outputs[0].size();
        int n2 = net2->outputs[0].size();
        int N = outputs.size();
        net1->d_outputs.resize(N);
        net2->d_outputs.resize(N);
        for (int t = 0; t < N; t++) {
            net1->d_outputs[t] = d_outputs[t].segment(0, n1);
            net2->d_outputs[t] = d_outputs[t].segment(n1, n2);
        }
        net1->backward();
        net2->backward();
        d_inputs.resize(N);
        for (int t = 0; t < N; t++) {
            d_inputs[t] = net1->d_inputs[t];
            d_inputs[t] += net2->d_inputs[t];
        }
    }
};

INetwork *make_Parallel() {
    return new Parallel();
}

namespace {
template <class NONLIN, class T>
inline Vec nonlin(T &a) {
    Vec result = a;
    NONLIN::f(result);
    return result;
}
template <class NONLIN, class T>
inline Vec yprime(T &a) {
    Vec result = Vec::Ones(a.size());
    NONLIN::df(result, a);
    return result;
}
template <class NONLIN, class T>
inline Vec xprime(T &a) {
    Vec result = Vec::Ones(a.size());
    Vec temp = a;
    NONLIN::f(temp);
    NONLIN::df(result, temp);
    return result;
}
template <typename F, typename T>
void each(F f, T &a) {
    f(a);
}
template <typename F, typename T, typename ... Args>
void each(F f, T &a, Args&&... args ...) {
    f(a);
    each(f, args ...);
}
}

struct LSTM : Network {
    // NB: verified gradients against Python implementation; this
    // code yields identical numerical results
#define SEQUENCES gix, gfx, gox, cix, gi, gf, go, ci, state
#define DSEQUENCES gierr, gferr, goerr, cierr, stateerr, outerr
#define WEIGHTS WGI, WGF, WGO, WCI
#define PEEPS WIP, WFP, WOP
#define DWEIGHTS DWGI, DWGF, DWGO, DWCI
#define DPEEPS DWIP, DWFP, DWOP
    Sequence source, SEQUENCES, sourceerr, DSEQUENCES;
    Mat WEIGHTS, DWEIGHTS;
    Vec PEEPS, DPEEPS;
    typedef SigmoidNonlin F;
    typedef TanhNonlin G;
    typedef TanhNonlin H;
    Float gradient_clipping = 10.0;
    int ni, no, nf;
    LSTM() {
        name = "lstm";
    }
    int noutput() {
        return no;
    }
    int ninput() {
        return ni;
    }
    void postLoad() {
        no = WGI.rows();
        nf = WGI.cols();
        assert(nf > no);
        ni = nf-no-1;
    }
    void init(int no, int ni) {
        int nf = 1+ni+no;
        this->ni = ni;
        this->no = no;
        this->nf = nf;
        each([no, nf](Mat &w) {
                 w = Mat::Random(no, nf) * 0.01;
             }, WEIGHTS);
        each([no](Vec &w) {
                 w = Vec::Random(no) * 0.01;
             }, PEEPS);
        clearUpdates();
    }
    void clearUpdates() {
        each([this](Mat &d) { d = Mat::Zero(no, nf); }, DWEIGHTS);
        each([this](Vec &d) { d = Vec::Zero(no); }, DPEEPS);
    }
    void resize(int N) {
        each([N](Sequence &s) {
                 s.resize(N);
                 for (int t = 0; t < N; t++) s[t].setConstant(NAN);
             }, source, sourceerr, outputs, SEQUENCES, DSEQUENCES);
        assert(source.size() == N);
        assert(gix.size() == N);
        assert(goerr.size() == N);
    }
#define A array()
    void forward() {
        int N = inputs.size();
        resize(N);
        for (int t = 0; t < N; t++) {
            source[t].resize(nf);
            source[t](0) = 1;
            source[t].segment(1, ni) = inputs[t];
            if (t == 0) source[t].segment(1+ni, no).setConstant(0);
            else source[t].segment(1+ni, no) = outputs[t-1];
            gix[t] = WGI * source[t];
            gfx[t] = WGF * source[t];
            gox[t] = WGO * source[t];
            cix[t] = WCI * source[t];
            if (t > 0) {
                gix[t].A += WIP.A * state[t-1].A;
                gfx[t].A += WFP.A * state[t-1].A;
            }
            gi[t] = nonlin<F>(gix[t]);
            gf[t] = nonlin<F>(gfx[t]);
            ci[t] = nonlin<G>(cix[t]);
            state[t] = ci[t].A * gi[t].A;
            if (t > 0) {
                state[t].A += gf[t].A * state[t-1].A;
                gox[t].A += WOP.A * state[t].A;
            }
            go[t] = nonlin<F>(gox[t]);
            outputs[t] = nonlin<H>(state[t]).A * go[t].A;
        }
    }
    void backward() {
        int N = inputs.size();
        d_inputs.resize(N);
        for (int t = N-1; t >= 0; t--) {
            outerr[t] = d_outputs[t];
            if (t < N-1) outerr[t] += sourceerr[t+1].segment(1+ni, no);
            goerr[t] = yprime<F>(go[t]).A * nonlin<H>(state[t]).A * outerr[t].A;
            stateerr[t] = xprime<H>(state[t]).A * go[t].A * outerr[t].A;
            stateerr[t].A += goerr[t].A * WOP.A;
            if (t < N-1) {
                stateerr[t].A += gferr[t+1].A * WFP.A;
                stateerr[t].A += gierr[t+1].A * WIP.A;
                stateerr[t].A += stateerr[t+1].A * gf[t+1].A;
            }
            if (t > 0) gferr[t] = yprime<F>(gf[t]).A * stateerr[t].A * state[t-1].A;
            gierr[t] = yprime<F>(gi[t]).A * stateerr[t].A * ci[t].A;
            cierr[t] = yprime<G>(ci[t]).A * stateerr[t].A * gi[t].A;
            sourceerr[t] = WGI.transpose() * gierr[t];
            if (t > 0) sourceerr[t] += WGF.transpose() * gferr[t];
            sourceerr[t] += WGO.transpose() * goerr[t];
            sourceerr[t] += WCI.transpose() * cierr[t];
            d_inputs[t] = sourceerr[t].segment(1, ni);
        }
        if (gradient_clipping > 0 || gradient_clipping < 999) {
            gradient_clip(gierr, gradient_clipping);
            gradient_clip(gferr, gradient_clipping);
            gradient_clip(goerr, gradient_clipping);
            gradient_clip(cierr, gradient_clipping);
        }
        for (int t = 0; t < N; t++) {
            if (t > 0) DWIP.A += gierr[t].A * state[t-1].A;
            if (t > 0) DWFP.A += gferr[t].A * state[t-1].A;
            DWOP.A += goerr[t].A * state[t].A;
            DWGI += gierr[t] * source[t].transpose();
            if (t > 0) DWGF += gferr[t] * source[t].transpose();
            DWGO += goerr[t] * source[t].transpose();
            DWCI += cierr[t] * source[t].transpose();
        }
        if (no_update) return;
        update();
        applyMomentum(momentum);
    }
#undef A
    void gradient_clip(Sequence &s, Float m=1.0) {
        for (int t = 0; t < s.size(); t++) {
            s[t] = s[t].unaryExpr([m](Float x) { return x > m ? m : x < -m ? -m : x; });
        }
    }
    void update() {
        WGI += lr * DWGI;
        WGF += lr * DWGF;
        WGO += lr * DWGO;
        WCI += lr * DWCI;
        WIP += lr * DWIP;
        WFP += lr * DWFP;
        WOP += lr * DWOP;
    }
    void applyMomentum(Float r) {
        DWGI *= momentum;
        DWGF *= momentum;
        DWGO *= momentum;
        DWCI *= momentum;
        DWIP *= momentum;
        DWFP *= momentum;
        DWOP *= momentum;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".WGI", &WGI, &DWGI);
        f(prefix+".WGF", &WGF, &DWGF);
        f(prefix+".WGO", &WGO, &DWGO);
        f(prefix+".WCI", &WCI, &DWCI);
        f(prefix+".WIP", &WIP, &DWIP);
        f(prefix+".WFP", &WFP, &DWFP);
        f(prefix+".WOP", &WOP, &DWOP);
    }
    virtual void mystates(string prefix, StateFun f) {
        f(prefix+".inputs", &inputs);
        f(prefix+".d_inputs", &d_inputs);
        f(prefix+".outputs", &outputs);
        f(prefix+".d_outputs", &d_outputs);
        f(prefix+".state", &state);
        f(prefix+".stateerr", &stateerr);
        f(prefix+".gi", &gi);
        f(prefix+".gierr", &gierr);
        f(prefix+".go", &go);
        f(prefix+".goerr", &goerr);
        f(prefix+".gf", &gf);
        f(prefix+".gferr", &gferr);
        f(prefix+".ci", &ci);
        f(prefix+".cierr", &cierr);
    }
};

INetwork *make_LSTM() {
    return new LSTM();
}

struct MLP : Network {
    LogregLayer l1, l2;
    MLP() {
        name = "mlp";
    }
    void init(int no, int nh, int ni) {
        l1.init(nh, ni);
        l2.init(no, nh);
    }
    void forward() {
        l1.inputs = inputs;
        l1.forward();
        l2.inputs = l1.outputs;
        l2.forward();
        outputs = l2.outputs;
    }
    void setLearningRate(Float lr, Float momentum) {
        l1.setLearningRate(lr, momentum);
        l2.setLearningRate(lr, momentum);
    }
    void backward() {
        l2.d_outputs = d_outputs;
        l2.backward();
        l1.d_outputs = l2.d_inputs;
        l1.backward();
        d_inputs = l1.d_inputs;
    }
    void weights(const string &prefix, WeightFun f) {
        l1.weights(prefix+".l1", f);
        l2.weights(prefix+".l2", f);
    }
    virtual void states(string prefix, StateFun f) {
        f(prefix+".inputs", &inputs);
        f(prefix+".d_inputs", &d_inputs);
        f(prefix+".outputs", &outputs);
        f(prefix+".d_outputs", &d_outputs);
        l1.states(prefix+".l1", f);
        l2.states(prefix+".l2", f);
    }
};

INetwork *make_MLP() {
    return new MLP();
}

struct LSTM1 : Stacked {
    LSTM1() {
        name = "lstm1";
    }
    void init(int no, int nh, int ni) {
        shared_ptr<INetwork> fwd, logreg;
        fwd = make_shared<LSTM>();
        fwd->init(nh, ni);
        add(fwd);
        logreg = make_shared<SoftmaxLayer>();
        logreg->init(no, nh);
        add(logreg);
    }
};

INetwork *make_LSTM1() {
    return new LSTM1();
}

struct REVLSTM1 : Stacked {
    REVLSTM1() {
        name = "revlstm1";
    }
    void init(int no, int nh, int ni) {
        shared_ptr<INetwork> fwd, rev, logreg;
        fwd = make_shared<LSTM>();
        fwd->init(nh, ni);
        rev = make_shared<Reversed>();
        rev->add(fwd);
        add(rev);
        logreg = make_shared<SoftmaxLayer>();
        logreg->init(no, nh);
        add(logreg);
    }
};

INetwork *make_REVLSTM1() {
    return new REVLSTM1();
}

struct BIDILSTM : Stacked {
    BIDILSTM() {
        name = "bidilstm";
    }
    void init(int no, int nh, int ni) {
        shared_ptr<INetwork> fwd, bwd, parallel, reversed, logreg;
        fwd = make_shared<LSTM>();
        fwd->init(nh, ni);
        bwd = make_shared<LSTM>();
        bwd->init(nh, ni);
        reversed = make_shared<Reversed>();
        reversed->add(bwd);
        parallel = make_shared<Parallel>();
        parallel->add(fwd);
        parallel->add(reversed);
        add(parallel);
        logreg = make_shared<SoftmaxLayer>();
        logreg->init(no, 2*nh);
        add(logreg);
    }
};

INetwork *make_BIDILSTM() {
    return new BIDILSTM();
}

inline Float log_add(Float x, Float y) {
    if (abs(x-y) > 10) return fmax(x, y);
    return log(exp(x-y)+1) + y;
}

inline Float log_mul(Float x, Float y) {
    return x+y;
}

void forward_algorithm(Mat &lr, Mat &lmatch, double skip) {
    int n = lmatch.rows(), m = lmatch.cols();
    lr.resize(n, m);
    Vec v(m), w(m);
    for (int j = 0; j < m; j++) v(j) = skip * j;
    for (int i = 0; i < n; i++) {
        w.segment(1, m-1) = v.segment(0, m-1);
        w(0) = skip * i;
        for (int j = 0; j < m; j++) {
            Float same = log_mul(v(j), lmatch(i, j));
            Float next = log_mul(w(j), lmatch(i, j));
            v(j) = log_add(same, next);
        }
        lr.row(i) = v;
    }
}

void forwardbackward(Mat &both, Mat &lmatch) {
    Mat lr;
    forward_algorithm(lr, lmatch);
    Mat rlmatch = lmatch;
    rlmatch = rlmatch.rowwise().reverse().eval();
    rlmatch = rlmatch.colwise().reverse().eval();
    Mat rl;
    forward_algorithm(rl, rlmatch);
    rl = rl.colwise().reverse().eval();
    rl = rl.rowwise().reverse().eval();
    both = lr + rl;
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Sequence &targets) {
    double lo = 1e-5;
    int n1 = outputs.size();
    int n2 = targets.size();
    int nc = targets[0].size();

    // compute log probability of state matches
    Mat lmatch;
    lmatch.resize(n1, n2);
    for (int t1 = 0; t1 < n1; t1++) {
        Vec out = outputs[t1].cwiseMax(lo);
        out /= out.sum();
        for (int t2 = 0; t2 < n2; t2++) {
            double value = out.transpose() * targets[t2];
            lmatch(t1, t2) = log(value);
        }
    }
    // compute unnormalized forward backward algorithm
    Mat both;
    forwardbackward(both, lmatch);

    // compute normalized state probabilities
    Mat epath = (both.array() - both.maxCoeff()).unaryExpr(ptr_fun(limexp));
    for (int j = 0; j < n2; j++) {
        double l = epath.col(j).sum();
        epath.col(j) /= l == 0 ? 1e-9 : l;
    }
    debugmat = epath;

    // compute posterior probabilities for each class and normalize
    Mat aligned;
    aligned.resize(n1, nc);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < nc; j++) {
            double total = 0.0;
            for (int k = 0; k < n2; k++) {
                double value = epath(i, k) * targets[k](j);
                total += value;
            }
            aligned(i, j) = total;
        }
    }
    for (int i = 0; i < n1; i++) {
        aligned.row(i) /= fmax(1e-9, aligned.row(i).sum());
    }

    // assign to outputs
    posteriors.resize(n1);
    for (int i = 0; i < n1; i++) {
        posteriors[i] = aligned.row(i);
    }
    assert(posteriors[0].size() == nc);
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Classes &targets) {
    int nclasses = outputs[0].size();
    Sequence stargets;
    stargets.resize(targets.size());
    for (int t = 0; t < stargets.size(); t++) {
        stargets[t].resize(nclasses);
        stargets[t].fill(0);
        stargets[t](targets[t]) = 1.0;
    }
    ctc_align_targets(posteriors, outputs, stargets);
}

void mktargets(Sequence &seq, Classes &transcript, int ndim) {
    seq.resize(2*transcript.size()+1);
    for (int t = 0; t < seq.size(); t++) {
        seq[t].setZero(ndim);
        if (t%2 == 1) seq[t](transcript[(t-1)/2]) = 1;
        else seq[t](0) = 1;
    }
}
}  // namespace ocropus

#ifdef LSTM_TEST
// We include the test cases in the source file because we want
// direct access to internal variables from the test cases.
#include "lstm_test.i"
#endif

