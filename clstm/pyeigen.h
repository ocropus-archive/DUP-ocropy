// -*- C++ -*-

#ifndef pyeigen__
#define pyeigen__

#include <zmqpp/zmqpp.hpp>
#include <string>
#include <stdarg.h>
#include <iostream>
#include <Eigen/Dense>

namespace pyeigen {
using namespace std;

inline string stringf(const char *format, ...) {
    static char buf[4096];
    va_list v;
    va_start(v, format);
    vsnprintf(buf, sizeof (buf), format, v);
    va_end(v);
    return string(buf);
}

struct PyServer {
    int mode = 0;  // -1=ignore, 0=uninit, 1=output
    zmqpp::context context;
    unique_ptr<zmqpp::socket> socket;
    void open(const char *where="tcp://127.0.0.1:9876") {
        if (string(where) == "none") {
            mode = -1;
            return;
        }
        socket.reset(new zmqpp::socket(context, zmqpp::socket_type::req));
        string addr = getenv("PYSERVER") ? getenv("PYSERVER") : where;
        cerr << "waiting for python server at " << addr << endl;
        socket->connect(addr.c_str());
        mode = 1;
        eval("print 'OK'");
        cerr << "connected" << endl;
        eval("from pylab import *");
        eval("ion()");
    }
    void setMode(int mode) {
        this->mode = mode;
    }
    string eval(string s) {
        if (mode < 0) return ""; else if (mode < 1) throw "uninitialized";
        zmqpp::message message;
        message << s;
        socket->send(message);
        socket->receive(message);
        string result;
        message >> result;
        return result;
    }
    string eval(string s, const float *a, int na) {
        if (mode < 0) return ""; else if (mode < 1) throw "uninitialized";
        string cmd;
        zmqpp::message message;
        message << cmd + s;
        message.add_raw((const char *)a, na*sizeof (float));
        socket->send(message);
        socket->receive(message);
        string response;
        message >> response;
        return response;
    }
    string eval(string s, const float *a, int na, const float *b, int nb) {
        if (mode < 0) return ""; else if (mode < 1) throw "uninitialized";
        string cmd;
        zmqpp::message message;
        message << cmd + s;
        message.add_raw((const char *)a, na*sizeof (float));
        message.add_raw((const char *)b, nb*sizeof (float));
        socket->send(message);
        socket->receive(message);
        string response;
        message >> response;
        return response;
    }
    string evalf(const char *format, ...) {
        static char buf[4096];
        va_list v;
        va_start(v, format);
        vsnprintf(buf, sizeof (buf), format, v);
        va_end(v);
        return eval(buf);
    }
    void clf() {
        eval("clf()");
    }
    void subplot(int rows, int cols, int n) {
        eval(stringf("subplot(%d,%d,%d)", rows, cols, n));
    }
    void plot(const Eigen::VectorXf &v, string extra="") {
        if (extra != "") extra = string(",")+extra;
        eval(stringf("plot(farg(1)%s)", extra.c_str()), &v(0), v.size());
    }
    void plot2(const Eigen::VectorXf &u, const Eigen::VectorXf &v, string extra="") {
        if (extra != "") extra = string(",")+extra;
        eval(stringf("plot(farg(1),farg(2)%s)", extra.c_str()),
             &u(0), u.size(),
             &v(0), v.size());
    }
    void imshow(const Eigen::MatrixXf &a, string extra="") {
        if (extra != "") extra = string(",")+extra;
        eval(stringf("imshow(farg2(1,%d,%d)%s)", a.cols(), a.rows(), extra.c_str()),
             &a(0, 0), a.rows()*a.cols());
    }
};

inline PyServer *make_PyServer() {
    return new PyServer();
}
}

#endif
