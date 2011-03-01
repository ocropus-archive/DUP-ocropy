#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

import glob
from distutils.core import setup, Extension

setup (name = 'ocropy',
       version = '0.1',
       author      = "Thomas Breuel",
       description = """Python bindings for OCRopus""",
       packages = ["ocrolib"],
       scripts = [i for i in glob.glob("ocropus-*[a-z]") if not i.endswith('.glade')] +
                 glob.glob("ocroex-*[a-z]") +
                 glob.glob("ocrotest-*[a-z]"),
       data_files=[('share/ocropus/gui', glob.glob("*.glade")),
                   ('share/ocropus/models', glob.glob("data/*model")),
                   ('share/ocropus/models', glob.glob("data/*.fst")),
                   ]
       )
