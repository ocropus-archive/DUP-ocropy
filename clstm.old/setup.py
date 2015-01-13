#!/usr/bin/env python

import os
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

include_dirs = ['/usr/include/eigen3','/usr/local/include'] + get_numpy_include_dirs()
swig_opts = ["-c++"] + ["-I" + d for d in include_dirs]
swiglib = os.popen("swig -swiglib").read()[:-1]

clstm = Extension('_clstm',
        libraries = ['hdf5_cpp','hdf5'],
        swig_opts = swig_opts,
        include_dirs = include_dirs,
        extra_compile_args = ['-std=c++11','-Wno-sign-compare', '-Wno-strict-prototypes'],
        sources=['clstm.i','clstm.cc'])

setup (name = 'clstm',
       version = '0.0',
       author      = "Thomas Breuel",
       description = """clstm library bindings""",
       ext_modules = [clstm],
       py_modules = ["clstm"])
