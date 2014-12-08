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

int main(int argc,char **argv) {
    Mat m = Mat::Random(17,34);

    unique_ptr<HDF5> h5(make_HDF5());
    h5->open("test_write.h5",true);
    h5->put(m,"m");
    h5.reset();
    return 0;
}

