#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

import glob,os,os.path
from distutils.core import setup, Extension
from distutils.command.install_data import install_data

from ocrolib import default
modelfiles = ["models/"+s for s in [default.model,default.space,default.ngraphs]]

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
