// -*- C++ -*-

#ifndef h5eigen__
#define h5eigen__

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <map>
#include "H5Cpp.h"

namespace h5eigen {
using namespace H5;
using namespace std;
using namespace Eigen;

struct HDF5 {
    shared_ptr<H5File> h5;
    void open(const char *name,bool rw=false) {
        if(rw) {
            h5.reset(new H5File(name,H5F_ACC_TRUNC));
        } else {
            h5.reset(new H5File(name,H5F_ACC_RDONLY));
        }
    }
    ~HDF5() {
        h5->close();
    }
    H5::PredType pred_type(int) { return PredType::NATIVE_INT; }
    H5::PredType pred_type(float) { return PredType::NATIVE_FLOAT; }
    H5::PredType pred_type(double) { return PredType::NATIVE_DOUBLE; }
    template <class T>
    void get(T &a,const char *name) {
        DataSet dataset = h5->openDataSet(name);
        DataSpace space = dataset.getSpace();
        hsize_t offset[] = {0,0,0,0,0,0};
        hsize_t count[] = {0,0,0,0,0,0};
        int rank = space.getSimpleExtentDims(count);
        if(rank==1) a.resize(count[0],1);
        else if(rank==2) a.resize(count[0],count[1]);
        else throw "unsupported rank";
        space.selectHyperslab(H5S_SELECT_SET, count, offset);
        DataSpace mem(rank, count);
        mem.selectHyperslab(H5S_SELECT_SET, count, offset);
        dataset.read(&a(0,0),pred_type(a(0,0)), mem, space);
    }
    template <class T>
    void get1d(T &a,const char *name) {
        DataSet dataset = h5->openDataSet(name);
        DataSpace space = dataset.getSpace();
        hsize_t offset[] = {0,0,0,0,0,0};
        hsize_t count[] = {0,0,0,0,0,0};
        int rank = space.getSimpleExtentDims(count);
        if(rank==1) a.resize(count[0]);
        else if(rank==2) a.resize(count[0]*count[1]);
        else if(rank==3) a.resize(count[0]*count[1]*count[2]);
        else if(rank==4) a.resize(count[0]*count[1]*count[2]*count[3]);
        else throw "unsupported rank";
        space.selectHyperslab(H5S_SELECT_SET, count, offset);
        DataSpace mem(rank, count);
        mem.selectHyperslab(H5S_SELECT_SET, count, offset);
        dataset.read(&a[0],pred_type(a[0]), mem, space);
    }
    template <class T>
    void put(T &a,const char *name,int rank=2) {
        DSetCreatPropList plist; // setFillValue, etc.
        hsize_t rows = a.rows();
        hsize_t cols = a.cols();
        hsize_t dim[] = {rows,cols,0,0,0,0};
        DataSpace fspace(rank,dim);
        DataSet dataset = h5->createDataSet(name, pred_type(a(0,0)), fspace, plist);
        hsize_t start[] = {0,0,0,0,0,0};
        hsize_t count[] = {rows,cols,0,0,0,0};
        DataSpace mspace(rank,dim);
        fspace.selectHyperslab(H5S_SELECT_SET, count, start);
        mspace.selectHyperslab(H5S_SELECT_SET, count, start);
        dataset.write(&a(0,0),pred_type(a(0,0)), mspace, fspace);
    }
    DataSet attrDataSet() {
        const char *name = "attributes";
        try {
            DataSet dataset = h5->openDataSet(name);
            return dataset;
        } catch(...) {}
        try {
            DSetCreatPropList plist; // setFillValue, etc.
            hsize_t dim[] = {1};
            DataSpace fspace(1,dim);
            DataSet dataset = h5->createDataSet(name, pred_type(0), fspace, plist);
            return dataset;
        } catch(...) {}
        throw "cannot open or create attrDataSet";
    }
    void setAttr(string name,string value) {
        DataSet dataset = attrDataSet();
        StrType strdatatype(PredType::C_S1, 256);
        DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
        H5std_string buffer(value);
        dataset.createAttribute(name,strdatatype,attr_dataspace)
            .write(strdatatype,buffer);
    }
    string getAttr(string name) {
        DataSet dataset;
        try {dataset = attrDataSet();}
        catch(FileIException e) { return ""; }
        StrType strdatatype(PredType::C_S1, 256);
        DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
        H5std_string buffer;
        dataset.createAttribute(name,strdatatype,attr_dataspace)
            .read(strdatatype,buffer);
        return buffer;
    }
    void getAttrs(map<string,string> &result) {
        DataSet dataset;
        try {dataset = attrDataSet();}
        catch(FileIException e) { return; }
        StrType strdatatype(PredType::C_S1, 256);
        DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
        H5std_string buffer;
        for(int i=0;i<dataset.getNumAttrs();i++) {
            Attribute a = dataset.openAttribute(i);
            string key = a.getName();
            a.read(strdatatype,buffer);
            result[key] = buffer;
        }
    }
};

inline HDF5 *make_HDF5() {
    return new HDF5();
}
}

#endif
