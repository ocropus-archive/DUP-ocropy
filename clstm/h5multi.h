// -*- C++ -*-

#ifndef h5multi__
#define h5multi__

#include <type_traits>
#include <memory>
#include <string>
#include <map>
#include "H5Cpp.h"
#include "multidim.h"

namespace h5multi {
using namespace H5;
using namespace std;
using namespace multidim;

struct HDF5 {
    shared_ptr<H5File> h5;
    void open(const string &name, bool rw=false) {
        if (rw) {
            h5.reset(new H5File(name, H5F_ACC_TRUNC));
        } else {
            h5.reset(new H5File(name, H5F_ACC_RDONLY));
        }
    }
    ~HDF5() {
        h5->close();
    }
    H5::PredType pred_type(int) {
        return PredType::NATIVE_INT;
    }
    H5::PredType pred_type(float) {
        return PredType::NATIVE_FLOAT;
    }
    H5::PredType pred_type(double) {
        return PredType::NATIVE_DOUBLE;
    }

    struct dims_ {
        int dims[8];
        operator int*() { return dims; }
    };
    inline static dims_ dims(int *p, int n) {
        dims_ result;
        for (int i = 0; i < 8; i++) result.dims[i] = i < n ? p[i] : 0;
        return result;
    }

    template <class T>
    void put(T &a, const string &name, int rank=2) {
        DSetCreatPropList plist;  // setFillValue, etc.
        hsize_t dims[8];
        for (int i = 0; i < 8; i++) dims[i] = a.dims[i];
        DataSpace fspace(a.rank(), dims);
        DataSet dataset = h5->createDataSet(name, pred_type(*a.data), fspace, plist);
        hsize_t start[] = {0, 0, 0, 0, 0, 0, 0, 0};
        hsize_t count[8];
        for (int i = 0; i < 8; i++) count[i] = a.dims[i];
        DataSpace mspace(rank, dims);
        fspace.selectHyperslab(H5S_SELECT_SET, count, start);
        mspace.selectHyperslab(H5S_SELECT_SET, count, start);
        dataset.write(a.data, pred_type(*a.data), mspace, fspace);
    }
    template <class T>
    void shape(mdarray<T> &a, const string &name) {
        DataSet dataset = h5->openDataSet(name);
        DataSpace space = dataset.getSpace();
        hsize_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};
        int rank = space.getSimpleExtentDims(count);
        a.resize(rank);
        for (int i = 0; count[i]; i++) a[i] = count[i];
    }
    template <class T>
    void get(mdarray<T> &a, const string &name) {
        DataSet dataset = h5->openDataSet(name);
        DataSpace space = dataset.getSpace();
        hsize_t offset[] = {0, 0, 0, 0, 0, 0, 0, 0};
        hsize_t count[] = {0, 0, 0, 0, 0, 0, 0, 0};
        int rank = space.getSimpleExtentDims(count);
        a.resize_(count);
        space.selectHyperslab(H5S_SELECT_SET, count, offset);
        DataSpace mem(rank, count);
        mem.selectHyperslab(H5S_SELECT_SET, count, offset);
        dataset.read(a.data, pred_type(*a.data), mem, space);
    }
    template <class T>
    void getrow(mdarray<T> &a, int index, const string &name) {
        DataSet dataset = h5->openDataSet(name);
        DataSpace fspace = dataset.getSpace();
        hsize_t start0[] = {0, 0, 0, 0, 0, 0, 0, 0};
        hsize_t dims[] = {0, 0, 0, 0, 0, 0, 0, 0};
        int rank = fspace.getSimpleExtentDims(dims);
        a.resize_(dims+1);
        hsize_t count[8];
        for (int i = 0; i < 8; i++) count[i] = dims[i];
        count[0] = 1;
        hsize_t start[] = {hsize_t(index), 0, 0, 0, 0, 0, 0, 0};
        DataSpace mspace(rank, count);
        fspace.selectHyperslab(H5S_SELECT_SET, count, start);
        mspace.selectHyperslab(H5S_SELECT_SET, count, start0);
        dataset.read(a.data, pred_type(*a.data), mspace, fspace);
    }
    template <class T>
    void getvlrow(T &a, int index, const string &name) {
        typedef typename remove_reference<decltype(a[0])>::type S;
        DataSet dataset = h5->openDataSet(name);
        DataSpace space = dataset.getSpace();
        hsize_t dims[] = {0, 0, 0, 0};
        int rank = space.getSimpleExtentDims(dims);
        assert(rank == 1);
        hsize_t start0[] = {0, 0};
        hsize_t start[] = {hsize_t(index), 0};
        hsize_t count[] = {1, 0};
        DataSpace fspace(1, dims);
        DataSpace mspace(1, count);
        fspace.selectHyperslab(H5S_SELECT_SET, count, start);
        mspace.selectHyperslab(H5S_SELECT_SET, count, start0);
        hvl_t vl[1];
        DataType ftype(pred_type(S(0)));
        VarLenType dtype(&ftype);
        dataset.read(vl, dtype, mspace, fspace);
        S *data = (S*)vl[0].p;
        int N = vl[0].len;
        a.resize(N);
        for (int i = 0; i < N; i++) a.data[i] = data[i];
        dataset.vlenReclaim(dtype, mspace, DSetMemXferPropList::DEFAULT, vl);
    }
    template <class T>
    void getarow(mdarray<T> &a, int index, const string &name) {
        mdarray<int> dims;
        shape(dims, name);
        if (dims.size() == 1) getvlrow(a, index, name);
        else getrow(a, index, name);
    }
    template <class T>
    void getdrow(mdarray<T> &a, int index, const string &name) {
        mdarray<int> dims;
        shape(dims, name);
        if (dims.size() == 1) {
            getvlrow(a, index, name);
            try {
                string sname(name);
                sname += "_dims";
                mdarray<int> ndims;
                getarow(ndims, index, sname.c_str());
                int ndims0[8];
                for (int i = 0; i < ndims.size(); i++) ndims0[i] = ndims[i];
                ndims0[ndims.size()] = 0;
                a.reshape_(ndims0);
            } catch(H5::Exception e) {
                // skip resizing if we can't get _dims, assume it's 1D
            }
        } else {
            getrow(a, index, name);
        }
    }
};

inline HDF5 *make_HDF5() {
    return new HDF5();
}
}

#endif
