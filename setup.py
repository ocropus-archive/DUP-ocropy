#!/usr/bin/env python

import sys
try:
    import scipy.ndimage
    import tables
except:
    print
    print "ERROR: import of scipy.ndimage or tables failed"
    print
    print "You may need to run ocropus-install-packages first."
    print
    sys.exit(1)

import glob,os,os.path
from distutils.core import setup, Extension
from distutils.command.install_data import install_data

from ocrolib import default
modelfiles = ["models/"+s for s in [default.model,default.space,default.ngraphs]]

for fname in modelfiles:
    if not os.path.exists(fname):
        print
        print "ERROR: cannot find",fname
        print
        print "You need to 'cd ./models' and run 'ocropus-download-models'"
        print
        sys.exit(1)

setup (name = 'ocropy',
       version = '0.5',
       author      = "Thomas Breuel",
       description = """Python bindings for OCRopus""",
       packages = ["ocrolib"],
       data_files=[('share/ocropus', glob.glob("*.glade")),
                   ('share/ocropus', modelfiles),
                   ],
       scripts = [i for i in glob.glob("ocropus-*[a-z5]") if not i.endswith('.glade')] +
                 glob.glob("ocroex-*[a-z]") +
                 glob.glob("ocrotest-*[a-z]") +
                 ["ocropus"],
       )
